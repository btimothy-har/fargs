import asyncio
import logging
from functools import wraps

import tiktoken
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger("fargs")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s [%(asctime)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def rate_limited_task(max_rate: int = 10, interval: int = 1):
    task_limiter = AsyncLimiter(int(max_rate), int(interval))

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            async with task_limiter:
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def token_limited_task(
    encoder_model: str = "cl100k_base",
    max_tokens_per_minute: int = 100_000,
    max_requests_per_minute: int = 1_000,
):
    tokenizer = tiktoken.get_encoding(encoder_model)

    if int(max_tokens_per_minute) >= 1_000_000:
        seconds_per_million = (60 * 1_000_000) / int(max_tokens_per_minute)
        token_limiter = AsyncLimiter(1_000_000, seconds_per_million)
    else:
        token_limiter = AsyncLimiter(int(max_tokens_per_minute), 60)

    rate_limiter = AsyncLimiter(int(int(max_requests_per_minute) // 60), 1)

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            total_tokens = 0
            for arg in args:
                total_tokens += len(tokenizer.encode(str(arg)))
            for key, value in kwargs.items():
                total_tokens += len(tokenizer.encode(str(key)))
                total_tokens += len(tokenizer.encode(str(value)))

            async with rate_limiter:
                await token_limiter.acquire(total_tokens)
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def sequential_task(concurrent_tasks: int = 1):
    sem = asyncio.Semaphore(concurrent_tasks)

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            async with sem:
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


async def tqdm_iterable(iterable, desc: str, **kwargs):
    async for item in tqdm_asyncio(iterable, desc=desc, **kwargs):
        yield item


async def async_batch(items, batch_size: int):
    """
    Yield items in batches for async processing.

    Args:
        items: Iterable of items to process
        batch_size: Size of each batch

    Yields:
        List of items in the current batch
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
        await asyncio.sleep(0)


def sync_batch(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
