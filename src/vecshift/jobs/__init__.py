"""Job specs: the declarative description of a migration, shared by every interface."""

from vecshift.jobs.spec import JobError, JobSpec, load_job, qdrant_template, template

__all__ = ["JobError", "JobSpec", "load_job", "qdrant_template", "template"]
