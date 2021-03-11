import ray
import time
import multiprocessing
from tqdm import tqdm
from threading import Event
from typing import Tuple
from ray.actor import ActorHandle

# Start Ray.
#ray.init(num_cpus=multiprocessing.cpu_count())
    
#items = []

#it = ray.util.iter.from_items(items, num_shards=multiprocessing.cpu_count())
#for doc in it.gather_async():

		
@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        # await self.event.wait()
        # self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


######################################################################
# This is where the progress bar starts. You create one of these
# on the head node, passing in the expected total number of items,
# and an optional string description.
# Pass along the `actor` reference to any remote task,
# and if they complete ten
# tasks, they'll call `actor.update.remote(10)`.

# Back on the local node, once you launch your remote Ray tasks, call
# `print_until_done`, which will feed everything back into a `tqdm` counter.


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


#################################################################
# This is an example of a task that increments the progress bar.
# Note that this is a Ray Task, but it could very well
# be any generic Ray Actor.
#
@ray.remote
def sleep_then_increment(i: int, pba: ActorHandle) -> int:
    time.sleep(i / 10)
    pba.update.remote(1)
    return i


#################################################################
# Now you can run it and see what happens!
#


def run():
    ray.init(num_cpus=multiprocessing.cpu_count())
    num_ticks = 40
    pb = ProgressBar(num_ticks)
    actor = pb.actor
    # You can replace this with any arbitrary Ray task/actor.
    #tasks_pre_launch = [
    #    sleep_then_increment.remote(i, actor) for i in range(0, num_ticks)
    #]

    items = list(range(0, num_ticks))
    it = ray.util.iter.from_items(items, num_shards=multiprocessing.cpu_count())
    for i in it.gather_async():
        sleep_then_increment.remote(i, actor)

    pb.print_until_done()
    #tasks = ray.get(tasks_pre_launch)

#run()
	