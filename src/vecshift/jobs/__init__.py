"""Job specs: the declarative description of a migration, shared by every interface."""

from vecshift.jobs.spec import JobError, JobSpec, load_job, template

__all__ = ["JobError", "JobSpec", "load_job", "template"]
