import asyncio
from functools import wraps

import tiktoken
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio


def rate_limited_task(max_rate: int = 10, interval: int = 1):
    task_limiter = AsyncLimiter(int(max_rate), int(interval))

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            async with task_limiter:
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def token_limited_task(max_tokens: int = 100_000, interval: int = 60):
    task_limiter = AsyncLimiter(int(max_tokens), int(interval))

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            tokenizer = tiktoken.get_encoding("cl100k_base")

            total_tokens = 0
            for arg in args:
                total_tokens += len(tokenizer.encode(str(arg)))
            for key, value in kwargs.items():
                total_tokens += len(tokenizer.encode(str(key)))
                total_tokens += len(tokenizer.encode(str(value)))

            await task_limiter.acquire(total_tokens)
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
