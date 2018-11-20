#![feature(arbitrary_self_types, async_await, await_macro, fnbox, futures_api, pin)]

use std::collections::VecDeque;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;
use std::sync::{self, Arc, Mutex};
use std::task::{self, LocalWaker, local_waker_from_nonlocal, Poll};

/// A simple task executor for use in tests and demos.
///
/// A `MockExecutor`'s `spawn` method takes a unit `Future` and starts a task
/// running it. The executor keeps track of which tasks are awake (that is, need
/// to be polled), and which are asleep.
struct MockExecutor {
    /// Tasks which are ready to be polled.
    awake: Arc<TaskQueue>,

    /// Closures representing events that occur outside all tasks, like I/O
    /// completing. Whenever the 'awake' list is empty, we call all these
    /// objects. Doing so should wake up some task. If no task is awakened, then
    /// the executor is deadlocked, and exits.
    outside: Vec<Box<dyn FnMut()>>
}

struct TaskQueue(Mutex<VecDeque<Arc<TaskBox>>>);

struct TaskBox(Mutex<Option<Pin<Box<dyn Future<Output = ()> + Send + 'static>>>>);

struct Waker {
    task: sync::Arc<TaskBox>,
    queue: sync::Weak<TaskQueue>,
}

impl MockExecutor {
    fn new() -> MockExecutor {
        MockExecutor {
            awake: Arc::new(TaskQueue(Mutex::new(VecDeque::new()))),
            outside: vec![],
        }
    }

    fn add_outside<F>(&mut self, outside: F)
        where F: FnMut() + 'static
    {
        self.outside.push(Box::new(outside))
    }

    fn spawn<A>(&mut self, task: A)
        where A: Future<Output = ()> + Send + 'static
    {
        let task_box = TaskBox::new(task);
        self.awake.push_back(Arc::new(task_box))
    }

    fn run(&mut self) {
        while !self.awake.is_empty() {
            while let Some(task_box) = self.awake.pop_front() {
                let waker = Waker {
                    task: Arc::clone(&task_box),
                    queue: Arc::downgrade(&self.awake),
                };
                let local_waker = local_waker_from_nonlocal(Arc::new(waker));

                // If the task box we've dequeued is empty, that means the task
                // completed, but then someone tried to wake it up again, so we
                // can just ignore that case.
                if let Some(mut pinned_task) = {
                    let mut guard = task_box.0.lock().unwrap();
                    guard.take()
                } {
                    eprintln!("executor: polling task");
                    match pinned_task.as_mut().poll(&local_waker) {
                        Poll::Ready(()) => {
                            eprintln!("executor: task completed");
                            // This task has completed. Any future attempts to wake it
                            // should have no effect, so leave the task box empty.
                        },
                        Poll::Pending => {
                            eprintln!("executor: task still pending");
                            // This task is pending. Any calls to waker::wake should
                            // enqueue its box again, so put it back in its task box.
                            *task_box.0.lock().unwrap() = Some(pinned_task);
                        }
                    }
                }
            }

            // All tasks are asleep. Let the outside world have a chance to behave.
            for outside in self.outside.iter_mut() {
                outside();
            }
        }

        // We ran the loop until all tasks were either complete or asleep, and
        // the outside world didn't wake anything up, so we're done.
    }
}

impl TaskBox {
    /// Create a new `TaskBox` holding the given task, a `Future` of a `()`
    /// value.
    fn new<A: Future<Output = ()> + Send + 'static>(task: A) -> TaskBox {
        TaskBox(Mutex::new(Some(Pin::from(Box::new(task)))))
    }
}

impl TaskQueue {
    fn is_empty(&self) -> bool {
        self.0.lock().unwrap().is_empty()
    }

    fn push_back(&self, task_box: Arc<TaskBox>) {
        let mut guard = self.0.lock().unwrap();
        // If this task box is already enqueued, don't queue it again.
        if guard.iter().any(|arc| Arc::ptr_eq(arc, &task_box)) {
            return;
        }

        guard.push_back(task_box);
    }

    fn pop_front(&self) -> Option<Arc<TaskBox>> {
        let mut guard = self.0.lock().unwrap();
        guard.pop_front()
    }
}

impl task::Wake for Waker {
    fn wake(arc_self: &Arc<Waker>) {
        // If the executor is gone, this has no effect.
        let queue = if let Some(queue) = arc_self.queue.upgrade() { queue } else { return; };

        queue.push_back(arc_self.task.clone());
    }
}

/// A 'scriptable' future that checks the poll requests it receives against
/// a supplied list of expectations.
struct MockFuture<T>(Arc<Mutex<InnerMockFuture<T>>>);

impl<T> Clone for MockFuture<T> {
    fn clone(&self) -> MockFuture<T> {
        MockFuture(self.0.clone())
    }
}

struct InnerMockFuture<T> {
    name: String,

    expectations: Box<dyn Iterator<Item=MockPollExpectation<T>> + Send>,

    /// A history of the wakers that have polled us.
    wakers: Vec<task::Waker>,
}

/// An interaction a `MockFuture` should expect in the process of testing. The
/// `MockFuture` is driven by a sequence of these values, reacting as prescribed
/// if they are met and panicking if they are not.
#[derive(Debug)]
enum MockPollExpectation<T> {
    /// The `MockFuture` expects to be polled. It should return the given `Poll`
    /// value, and note the task doing the poll.
    Polled(Poll<T>),

    /// The `MockFuture` expects all tasks to be asleep, and should wake up the
    /// given set of tasks. A value of `i` indicates that the task that
    /// performed the `i`'th last poll request should be awoken (that is, `1`
    /// means to wake up the task that polled most recently).
    AllAsleep(Vec<usize>)
}

use MockPollExpectation::*;

impl<T: 'static> MockFuture<T> {
    fn new<I>(name: &str, expected: I) -> MockFuture<T>
        where I: IntoIterator<Item=MockPollExpectation<T>>,
              I::IntoIter: 'static + Send
    {
        MockFuture(Arc::new(Mutex::new(InnerMockFuture {
            name: name.to_owned(),
            expectations: Box::new(expected.into_iter()),
            wakers: vec![],
        })))
    }

    fn register_as_outside(&self, executor: &mut MockExecutor)
        where T: Debug
    {
        let this: MockFuture<T> = self.clone();
        executor.add_outside(move || {
            let mut guard = this.0.lock().unwrap();
            match guard.expectations.next() {
                Some(AllAsleep(to_wake)) => {
                    eprintln!("MockFuture '{}' is waking up {:?}", guard.name, to_wake);
                    let len = guard.wakers.len();
                    for index in to_wake {
                        guard.wakers[len - index].wake();
                    }
                },
                otherwise => {
                    panic!("MockFuture '{}' wasn't expecting everyone to be asleep, was expecting {:?}",
                           guard.name, otherwise);
                }
            }
        });
    }
}

impl<T: Debug> Future for MockFuture<T> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, lw: &LocalWaker) -> Poll<Self::Output> {
        let mut guard = self.0.lock().unwrap();
        guard.wakers.push(lw.clone().into_waker());
        match guard.expectations.next() {
            Some(Polled(p)) => {
                eprintln!("MockFuture '{}' got polled, returned {:?}", guard.name, p);
                p
            },
            otherwise => panic!("MockFuture '{}' got polled, but expected {:?}",
                                guard.name, otherwise),
        }
    }
}

async fn print<F>(label: &'static str, future: F)
    where F: Future,
          F::Output: Debug
{
    println!("{}: {:?}", label, await!(future));
}

fn main() {
    let mut exe = MockExecutor::new();
    let f = MockFuture::new("Stanley", vec![Polled(Poll::Pending),
                                            AllAsleep(vec![1]),
                                            Polled(Poll::Ready(42)),
                                            AllAsleep(vec![])]);
    f.register_as_outside(&mut exe);

    exe.spawn(print("suffer", f));
    exe.run();
}
