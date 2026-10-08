from taskiq import TaskiqScheduler
from taskiq.middlewares import SmartRetryMiddleware
from taskiq_redis import ListRedisScheduleSource, RedisStreamBroker

from airgap_rag.core.config import get_settings
from airgap_rag.jobs.errors import RetryableJobError

settings = get_settings()

retry_schedule_source = ListRedisScheduleSource(
    url=settings.redis_url,
    prefix="airgap-rag:retry",
)

broker = RedisStreamBroker(
    url=settings.redis_url,
    queue_name="airgap-rag:jobs",
    consumer_group_name="airgap-rag:workers",
    idle_timeout=(settings.job_stale_after_seconds + 60) * 1000,
).with_middlewares(
    SmartRetryMiddleware(
        default_retry_count=settings.job_max_attempts - 1,
        default_delay=settings.job_retry_delay_seconds,
        use_jitter=True,
        use_delay_exponent=True,
        max_delay_exponent=settings.job_retry_max_delay_seconds,
        types_of_exceptions=(RetryableJobError,),
        schedule_source=retry_schedule_source,
    )
)

scheduler = TaskiqScheduler(broker=broker, sources=[retry_schedule_source])
