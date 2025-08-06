# Try, Try Till Your Chatbot Succeeds: An Experiment For Reliable Web Scraping

*A deep dive into transforming a failing chatbot web scraper from 70% to 100% success rate using Python, Tenacity, and browser simulation*

---

## The Problem That Started It All

LLM powered chabots are smart... **real smart**, until questioned about something beyond the scope of their knowledge. Good thing for them (and us), the world wide web has (nearly) all the answers! What better source for additional context for your queries, right? Well I though so too.

So I write a simple web lookup function, run my chatbot and... the responses feel just as ignorant as before. That can't be right. My LLM is smart and looking up stuff online makes one smarter! (LOL)

I hack together a rudimentary evaluation script and, lo and behold!

```bash
$ python scripts/evaluate_web_lookup.py
Success rate: 92.5%
Failed fetches: 6
```

My chatbot isn't even smart enough to look up the web **~7.5% of the time!** In other words, a **92.5% rate of success.**

### The Original Implementation

Here's what my initial web fetching code looked like—simple, straightforward, but fatally flawed:

```python
async def _fetch_url(url: str, client: httpx.AsyncClient) -> str:
    """Fetch content from a single URL - ORIGINAL VERSION."""
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except httpx.HTTPStatusError as e:
        LOGGER.warning("Failed to fetch %s: %s", url, e)
        return ""
    except Exception as e:
        LOGGER.warning("Unexpected error fetching %s: %s", url, e)
        return ""
```

**What was wrong with this?**
- ❌ No retry logic for network hiccups
- ❌ No browser headers (screamed "BOT!") -> Get's blocked by many sites
- ❌ Failed immediately on any connection issue
- ❌ Gave up instantly when websites blocked us

## The Investigation Begins

A little bit of digging and I find:

- **1 ReadTimeout failure**: Washington Post timing out (10.96s)
- **3 HTTP 403 Forbidden errors**: Bloomberg, Product Hunt, and AP blocking scrapers
- **1 HTTP 401 Unauthorized**: Reuters rejecting our requests  
- **1 HTTP 400 Bad Request**: Twitter (redirected to x.com) blocking access

The most frustrating part? Major news sources like Reuters, Bloomberg, and AP were flat-out rejecting our requests with **401/403 Forbidden** errors. This wasn't just a network issue; this was deliberate anti-bot protection across multiple high-value sites.

## First Attempt: Manual Retry Logic

> What do we do when something doesn't work out the first time? We do it again and again and again...

I crafted a manual retry loop with exponential backoff:

```python
async def _fetch_url(url: str, client: httpx.AsyncClient, max_retries: int = 3) -> str:
    for attempt in range(max_retries + 1):
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.text
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                await asyncio.sleep(wait_time)
            else:
                return ""
```

**Result: Still 92.5% success rate.**

The retry logic worked for genuine network hiccups, but Reuters (and some others) was still giving us the cold shoulder. Those connection errors were being retried properly, but the blocking issues remained stubbornly persistent.

## Let's Get Tenacious: Using `tenacity` for Retry Logic

Now, I'm all for re-inventing the wheel but aside from the bragging rights, there isn't much more to that. Enter [`tenacity`](https://github.com/jd/tenacity), Python's premier retry library.

I refactored the manual retry logic to use Tenacity's elegant decorator pattern:

```python
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_log,
    after_log,
)

@retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)),
    stop=stop_after_attempt(4),  # 3 retries + 1 initial attempt
    wait=wait_exponential(multiplier=1, min=1, max=8),  # 1s, 2s, 4s, 8s
    before=before_log(LOGGER, logging.WARNING),
    after=after_log(LOGGER, logging.WARNING),
)
async def _fetch_url(url: str, client: httpx.AsyncClient) -> str:
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
        raise  # Let tenacity handle the retry
    except httpx.HTTPStatusError as e:
        LOGGER.warning("HTTP error for %s: %s", url, e)
        return ""
```

**Benefits of the decorator approach:**
- **Declarative configuration**: Retry behavior defined upfront
- **Built-in logging**: Automatic retry attempt logging with timing
- **Clean separation**: Function logic separate from retry logic
- **Less boilerplate**: No manual try/catch blocks for retry handling

**Result: Still 92.5% success rate.**

The code was cleaner and more maintainable, but Reuters was still playing hard to get.

## The Eureka Moment: Browser Headers

Modern websites implement sophisticated measures to detect and block automated web scrapers, i.e., "bots". Reuters wasn't just randomly blocking requests; it was specifically targeting requests that didn't look like real browsers.

Our original request looked like this:
```
GET https://www.reuters.com
User-Agent: python-httpx/0.25.0
```

To a sophisticated anti-bot system, this screams "I'm a robot!".

## The Browser Disguise

The solution? Have the scraping utility **behave like a browser**. Add all the bells and whistles of a modern browser to the headers and we get:

```python
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

response = await client.get(url, headers=headers, follow_redirects=True)
```

**Why these specific headers matter:**

- **User-Agent**: The browser's identity card
- **Accept headers**: What content types the "browser" can handle
- **Sec-Fetch-*** headers: Chrome's security headers that signal legitimate navigation
- **Connection: keep-alive**: Mimics persistent browser connections
- **Cache-Control**: Shows typical browser caching behavior

## The Moment of Truth

With my chatbot in disguise, I run the evaluation script and **BOOM!**

```bash
OVERVIEW:
Total URLs tested: 80
Successful fetches: 80
Failed fetches: 0
Success rate: 100.0%
Total evaluation time: 3.84s
Average time per URL: 0.05s

RELIABILITY ASSESSMENT:
🟢 EXCELLENT: Function is highly reliable
```

**100% success rate!** 

But the real victory was seeing this in the logs:
```
https://www.reuters.com ✓ SUCCESS 1.64s 1,096,546b
```

Reuters had finally accepted our scraper as a legitimate visitor, serving up over 1MB of content instead of blocking our requests.

### The Final Version

```python
@tenacity.retry(
    retry=tenacity.retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)),
    stop=tenacity.stop_after_attempt(4),  # 3 retries + 1 initial attempt
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=8),  # 1s, 2s, 4s, 8s
    before=tenacity.before_log(LOGGER, logging.WARNING),
    after=tenacity.after_log(LOGGER, logging.WARNING),
)
async def _fetch_url(url: str, client: httpx.AsyncClient) -> str:
    """Fetch content from a single URL - FINAL VERSION."""
    # Headers to appear more like a legitimate browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }

    try:
        response = await client.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
        LOGGER.warning("Error fetching %s: %s", url, e)
        raise  # Let tenacity handle the retry
    except httpx.HTTPStatusError as e:
        LOGGER.warning("HTTP error for %s: %s", url, e)
        return ""
    except Exception as e:
        LOGGER.warning("Unexpected error fetching %s: %s", url, e)
        return ""
```

**What made this version bulletproof?**
- ✅ Tenacity decorator with smart exponential backoff
- ✅ Complete browser header simulation (looks 100% legitimate)
- ✅ Selective retry only for connection errors
- ✅ Built-in logging and observability
- ✅ Clean separation of retry logic from business logic

## Reliability Gains

Let's see what this improvement meant in practice with real test results:

### 🔴 BEFORE: The Failing Original

```bash
$ python scripts/evaluate_web_lookup.py

OVERVIEW:
Total URLs tested: 80
Successful fetches: 74
Failed fetches: 6
Success rate: 92.5%
Total evaluation time: 11.28s
Average time per URL: 0.14s

ERROR BREAKDOWN:
  ReadTimeout: 1 (1.2%)
  HTTP 403 Forbidden: 3 (3.8%)
  HTTP 401 Unauthorized: 1 (1.2%)
  HTTP 400 Bad Request: 1 (1.2%)

DETAILED RESULTS:
URL                                Status    Content    Error
------------------------------------------------------------
https://www.twitter.com           ❌ FAILED    -        400 Bad Request
https://www.reuters.com           ❌ FAILED    -        401 Unauthorized
https://www.washingtonpost.com    ❌ FAILED    -        ReadTimeout
https://www.ap.org                ❌ FAILED    -        403 Forbidden
https://www.bloomberg.com         ❌ FAILED    -        403 Forbidden
https://www.producthunt.com       ❌ FAILED    -        403 Forbidden
```

**What went wrong:**
- **403 Forbidden errors**: Bloomberg, Product Hunt, AP blocked scrapers outright
- **401 Unauthorized**: Reuters rejected requests with authentication error
- **400 Bad Request**: Twitter (x.com redirect) blocked malformed requests  
- **ReadTimeout**: Washington Post took too long (10.96s timeout)
- **Anti-bot protection**: Sophisticated detection across major news sites
- **Bot-like headers** triggered security systems systematically

### 🟢 AFTER: The Bulletproof Final Version

```bash
$ python scripts/evaluate_web_lookup.py

OVERVIEW:
Total URLs tested: 80
Successful fetches: 80
Failed fetches: 0
Success rate: 100.0%
Total evaluation time: 3.84s
Average time per URL: 0.05s

DETAILED RESULTS:
URL                                Status     Content      Notes
----------------------------------------------------------------
https://www.reuters.com           ✅ SUCCESS  1,096,546b  Now works!
https://www.twitter.com           ✅ SUCCESS    268,816b  Fixed!
https://www.washingtonpost.com    ✅ SUCCESS  2,116,715b  No timeout!
https://www.bloomberg.com         ✅ SUCCESS    102,211b  Unblocked!
https://www.producthunt.com       ✅ SUCCESS     26,702b  Success!
https://www.google.com            ✅ SUCCESS     50,786b  
https://www.github.com            ✅ SUCCESS    553,355b  
https://docs.python.org/3/        ✅ SUCCESS     17,828b  
https://api.github.com            ✅ SUCCESS      2,396b  

RELIABILITY ASSESSMENT:
🟢 EXCELLENT: Function is highly reliable
```

**The major gains:**
- **Reuters now serves 1.1MB** instead of blocking
- **Twitter/Bloomberg unblocked** with proper headers
- **Washington Post timeout resolved** (10.96s → 1.58s)
- **100% success rate** across all 80 test URLs
- **Browser simulation** bypassed all anti-bot systems

## The Bigger Picture

This improvement directly enhanced my chatbot's ability to process web content from user messages. When users share URLs in chat, the chatbot can now reliably:

- ✅ Fetch content from major news sites (Reuters, BBC, CNN)
- ✅ Process technical documentation (Python docs, Streamlit docs)  
- ✅ Handle social media links (Reddit, Medium)
- ✅ Work with APIs and testing services

**UX Improvement:**
- **Better answers**: More complete knowledge base from web content
- **User trust**: Consistent behavior when processing URLs

## What's Next?

With 100% reliability achieved, future enhancements could include:

- **Header rotation**: Random User-Agent selection from a pool
- **Rate limiting**: Per-domain request throttling
- **Caching**: Avoid refetching recently accessed content
- **Circuit breakers**: Fail-fast for consistently problematic domains
