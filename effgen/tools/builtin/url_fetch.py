"""
URL Fetch tool for retrieving webpage content.

Fetches and extracts text from web pages using requests + BeautifulSoup
(both free/open source). Falls back to stdlib urllib if packages unavailable.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)


def _get_user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__} (URL Fetch Tool)"

logger = logging.getLogger(__name__)

# Well-known cloud metadata endpoints (covered by the range checks below too,
# listed explicitly so the failure message is unambiguous).
_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254", "100.100.100.200"}


def _is_blocked_ip(addr: ipaddress._BaseAddress) -> bool:
    """Return True for any non-public address an SSRF would target."""
    # Normalise IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) to its IPv4 form.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if str(addr) in _METADATA_IPS:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


class _BlockedURLError(ValueError):
    """Raised when a URL targets a non-public/internal address."""


def _build_no_redirect_opener():
    """Build a urllib opener that surfaces 3xx as HTTPError instead of
    silently following them, so each hop can be re-validated by the caller."""
    from urllib.error import HTTPError
    from urllib.request import HTTPRedirectHandler, build_opener

    class _NoRedirect(HTTPRedirectHandler):
        def http_error_301(self, req, fp, code, msg, headers):
            raise HTTPError(req.full_url, code, msg, headers, fp)

        http_error_302 = http_error_301
        http_error_303 = http_error_301
        http_error_307 = http_error_301
        http_error_308 = http_error_301

    return build_opener(_NoRedirect)


class _SimpleHTMLTextExtractor(HTMLParser):
    """Simple HTML to text converter using stdlib."""

    SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

    def __init__(self):
        super().__init__()
        self._text_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag.lower() in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._text_parts.append(data)

    def get_text(self) -> str:
        text = " ".join(self._text_parts)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


class URLFetchTool(BaseTool):
    """
    URL content fetching tool.

    Fetches webpage content and extracts readable text.

    Features:
    - Fetch and extract text from web pages
    - Configurable max content length
    - Domain allowlist/blocklist
    - Timeout control
    - Uses requests + beautifulsoup4 if available, falls back to urllib

    Note:
        This tool makes HTTP requests to external websites.
        Configure allowed_domains to restrict access.
    """

    def __init__(
        self,
        allowed_domains: set[str] | None = None,
        blocked_domains: set[str] | None = None,
        max_content_length: int = 10000,
        timeout: int = 15,
        allow_private: bool = False,
        max_redirects: int = 5,
    ):
        """
        Initialize the URL Fetch tool.

        Args:
            allowed_domains: If set, only fetch from these domains.
            blocked_domains: Domains to never fetch from.
            max_content_length: Max characters to return (default: 10000).
            timeout: Request timeout in seconds (default: 15).
            allow_private: Allow fetching private/loopback/link-local/metadata
                addresses. Default ``False`` blocks them (SSRF protection); set
                ``True`` only when you intentionally fetch internal services.
            max_redirects: Maximum number of redirects to follow; each hop's
                destination is re-validated against the SSRF rules.
        """
        super().__init__(
            metadata=ToolMetadata(
                name="url_fetch",
                description=(
                    "Fetch a webpage and extract its text content. "
                    "Returns the readable text from a URL. "
                    "Use this to get information from specific web pages."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="url",
                        type=ParameterType.STRING,
                        description="The URL to fetch (must start with http:// or https://)",
                        required=True,
                        min_length=8,
                    ),
                    ParameterSpec(
                        name="extract_links",
                        type=ParameterType.BOOLEAN,
                        description="Also extract links from the page",
                        required=False,
                        default=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
                timeout_seconds=timeout,
                tags=["url", "web", "fetch", "http", "scraping"],
                examples=[
                    {
                        "url": "https://example.com",
                        "output": {"title": "Example Domain", "text": "This domain is for use in..."},
                    },
                ],
            )
        )

        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains or set()
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.allow_private = allow_private
        self.max_redirects = max_redirects

        logger.info(
            "\u2139\ufe0f  URL Fetch tool makes HTTP requests to external websites.\n"
            "    Private/loopback/link-local/metadata addresses are blocked by "
            "default; set allow_private=True to override."
        )

    def _check_address_safe(self, url: str) -> None:
        """Resolve the host and reject non-public addresses (SSRF guard).

        Validates every IP the hostname resolves to, so a public name that maps
        to an internal address is rejected too. Called for the original URL and
        for each redirect hop.
        """
        if self.allow_private:
            return
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            raise _BlockedURLError("URL has no host")

        # A bare IP literal can be checked directly without DNS.
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            if _is_blocked_ip(literal):
                raise _BlockedURLError(
                    f"Refusing to fetch internal address '{host}' (SSRF "
                    "protection). Set allow_private=True to override."
                )
            return

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            raise _BlockedURLError(f"Cannot resolve host '{host}': {e}")
        for info in infos:
            ip_str = info[4][0]
            try:
                addr = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if _is_blocked_ip(addr):
                raise _BlockedURLError(
                    f"Refusing to fetch '{host}': it resolves to non-public "
                    f"address {ip_str} (SSRF protection). Set allow_private=True "
                    "to override."
                )

    def _validate_url(self, url: str) -> str:
        """Validate and normalize URL (scheme, domain lists, SSRF guard)."""
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)
        domain = parsed.hostname or ""

        if self.allowed_domains and domain not in self.allowed_domains:
            raise ValueError(f"Domain '{domain}' not in allowed domains list")
        if domain in self.blocked_domains:
            raise ValueError(f"Domain '{domain}' is blocked")

        self._check_address_safe(url)
        return url

    def _extract_with_beautifulsoup(self, html: str) -> tuple:
        """Extract text using BeautifulSoup."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string if soup.title else ""
        text = soup.get_text(separator="\n", strip=True)

        links = []
        for a in soup.find_all("a", href=True):
            links.append({"text": a.get_text(strip=True), "href": a["href"]})

        return title, text, links

    def _extract_with_stdlib(self, html: str) -> tuple:
        """Extract text using stdlib HTML parser."""
        extractor = _SimpleHTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()

        # Try to find title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        # Extract links
        links = []
        for m in re.finditer(r'<a\s+[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
            links.append({"text": re.sub(r'<[^>]+>', '', m.group(2)).strip(), "href": m.group(1)})

        return title, text, links

    def _fetch_with_requests(self, requests, url: str) -> str:
        """Fetch with manual, re-validated redirect following."""
        headers = {"User-Agent": _get_user_agent()}
        current = url
        with requests.Session() as session:
            for _ in range(self.max_redirects + 1):
                self._check_address_safe(current)
                resp = session.get(
                    current,
                    timeout=self.timeout,
                    headers=headers,
                    allow_redirects=False,
                )
                if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        break
                    current = urljoin(current, location)
                    continue
                resp.raise_for_status()
                return resp.text
            raise ValueError(f"Too many redirects (>{self.max_redirects})")

    def _fetch_with_urllib(self, url: str) -> str:
        """Stdlib fallback: validate each redirect hop before following it."""
        from urllib.error import HTTPError
        from urllib.request import Request

        headers = {"User-Agent": _get_user_agent()}
        current = url
        for _ in range(self.max_redirects + 1):
            self._check_address_safe(current)
            req = Request(current, headers=headers, method="GET")
            try:
                # Block urllib's automatic redirect handling by treating 3xx as
                # terminal, then re-validate and follow manually.
                opener = _build_no_redirect_opener()
                with opener.open(req, timeout=self.timeout) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except HTTPError as e:
                if e.code in (301, 302, 303, 307, 308):
                    location = e.headers.get("Location")
                    if not location:
                        raise ValueError("Redirect without Location header")
                    current = urljoin(current, location)
                    continue
                raise
        raise ValueError(f"Too many redirects (>{self.max_redirects})")

    async def _execute(
        self,
        url: str,
        extract_links: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Fetch URL and extract text."""
        url = self._validate_url(url)

        # Try requests first, fall back to urllib. Redirects are followed
        # manually so every hop is re-checked against the SSRF guard (a public
        # URL must not be able to bounce us onto an internal address).
        html = ""
        try:
            import requests
            html = self._fetch_with_requests(requests, url)
        except ImportError:
            html = self._fetch_with_urllib(url)

        # Extract text
        try:
            title, text, links = self._extract_with_beautifulsoup(html)
        except ImportError:
            title, text, links = self._extract_with_stdlib(html)

        # Truncate
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "\n... (content truncated)"

        result = {
            "url": url,
            "title": title,
            "text": text,
            "content_length": len(text),
        }

        if extract_links:
            result["links"] = links[:50]  # Limit links

        return result
