from abc import ABC, abstractmethod


class Scheduler(ABC):

    @abstractmethod
    def submit(self, script, environment=None):
        pass

    @abstractmethod
    def wait(self, job_id):
        pass