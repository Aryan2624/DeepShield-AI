import ipaddress
import math
import re

from collections import Counter
from urllib.parse import urlparse


SUSPICIOUS_KEYWORDS = {
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "secure",
    "update",
    "bank",
    "password",
    "credential",
    "confirm",
    "wallet",
    "payment",
    "invoice",
    "bonus",
    "reward",
    "free",
    "suspend",
    "unlock",
    "recover",
    "reset",
    "auth",
    "authentication",
}


SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "buff.ly",
    "is.gd",
    "cutt.ly",
    "tiny.cc",
    "rebrand.ly",
    "shorturl.at",
}


SUSPICIOUS_EXTENSIONS = {
    ".exe",
    ".scr",
    ".bat",
    ".cmd",
    ".msi",
    ".apk",
    ".jar",
    ".zip",
    ".rar",
    ".7z",
    ".ps1",
    ".vbs",
}


def prepare_url_for_parsing(url: str) -> str:
    """
    Add a temporary scheme when the URL has no scheme.

    The original URL remains unchanged for lexical features.
    """

    url = str(url).strip()

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        url,
    ):
        return "http://" + url

    return url


def safe_parse_url(url: str):
    """
    Parse a URL without allowing malformed URLs to crash preprocessing.

    Returns:
        parsed_url
        hostname
        parse_error

    parse_error:
        0 -> parsed normally
        1 -> malformed URL required fallback handling
    """

    prepared_url = prepare_url_for_parsing(url)

    try:
        parsed_url = urlparse(prepared_url)

        hostname = (
            parsed_url.hostname or ""
        ).lower()

        return (
            parsed_url,
            hostname,
            0,
        )

    except (ValueError, UnicodeError):
        pass


    # Some malformed URLs contain invalid square brackets.
    # Python may interpret these as broken IPv6 addresses.
    sanitized_url = (
        prepared_url
        .replace("[", "%5B")
        .replace("]", "%5D")
    )

    try:
        parsed_url = urlparse(
            sanitized_url
        )

        hostname = (
            parsed_url.hostname or ""
        ).lower()

        return (
            parsed_url,
            hostname,
            1,
        )

    except (ValueError, UnicodeError):
        return (
            None,
            "",
            1,
        )


def calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy.
    """

    if not text:
        return 0.0

    counts = Counter(text)

    length = len(text)

    entropy = 0.0

    for count in counts.values():

        probability = (
            count / length
        )

        entropy -= (
            probability
            * math.log2(probability)
        )

    return entropy


def contains_ip_address(hostname: str) -> int:
    """
    Detect IPv4 or IPv6 hostnames.
    """

    if not hostname:
        return 0

    clean_hostname = (
        hostname.strip("[]")
    )

    try:
        ipaddress.ip_address(
            clean_hostname
        )

        return 1

    except ValueError:
        return 0


def count_subdomains(hostname: str) -> int:
    """
    Estimate subdomain count.

    Examples:
        example.com -> 0
        login.example.com -> 1

    IP addresses always return 0.
    """

    if not hostname:
        return 0

    hostname = (
        hostname
        .lower()
        .strip(".")
    )

    if contains_ip_address(
        hostname
    ):
        return 0

    if hostname.startswith("www."):
        hostname = hostname[4:]

    labels = [
        part
        for part in hostname.split(".")
        if part
    ]

    if len(labels) <= 2:
        return 0

    return len(labels) - 2


def contains_shortener(hostname: str) -> int:
    """
    Detect known URL-shortening domains.
    """

    hostname = hostname.lower()

    for domain in SHORTENER_DOMAINS:

        if (
            hostname == domain
            or hostname.endswith(
                "." + domain
            )
        ):
            return 1

    return 0


def count_suspicious_keywords(url: str) -> int:
    """
    Count suspicious security-related keywords.
    """

    lower_url = url.lower()

    count = 0

    for keyword in SUSPICIOUS_KEYWORDS:

        if keyword in lower_url:
            count += 1

    return count


def has_suspicious_extension(path: str) -> int:
    """
    Detect potentially dangerous downloadable extensions.
    """

    lower_path = path.lower()

    for extension in SUSPICIOUS_EXTENSIONS:

        if lower_path.endswith(
            extension
        ):
            return 1

    return 0


def extract_url_features(url: str) -> dict:
    """
    Extract numerical lexical cybersecurity features from one URL.
    """

    if url is None:
        url = ""

    url = str(url).strip()

    (
        parsed_url,
        hostname,
        parse_error,
    ) = safe_parse_url(url)


    if parsed_url is not None:

        path = (
            parsed_url.path or ""
        )

        query = (
            parsed_url.query or ""
        )

        scheme = (
            parsed_url.scheme or ""
        ).lower()

    else:

        path = ""
        query = ""
        scheme = ""


    # -----------------------------------
    # Length features
    # -----------------------------------

    url_length = len(url)

    hostname_length = len(hostname)

    path_length = len(path)

    query_length = len(query)


    # -----------------------------------
    # Character counts
    # -----------------------------------

    digit_count = sum(
        character.isdigit()
        for character in url
    )

    letter_count = sum(
        character.isalpha()
        for character in url
    )

    special_character_count = sum(
        not character.isalnum()
        for character in url
    )

    non_ascii_count = sum(
        ord(character) > 127
        for character in url
    )


    # -----------------------------------
    # Ratios
    # -----------------------------------

    safe_length = max(
        url_length,
        1,
    )

    digit_ratio = (
        digit_count
        / safe_length
    )

    letter_ratio = (
        letter_count
        / safe_length
    )

    special_character_ratio = (
        special_character_count
        / safe_length
    )

    non_ascii_ratio = (
        non_ascii_count
        / safe_length
    )


    # -----------------------------------
    # Security indicators
    # -----------------------------------

    uses_https = int(
        scheme == "https"
    )

    has_ip = contains_ip_address(
        hostname
    )

    subdomain_count = (
        count_subdomains(
            hostname
        )
    )

    suspicious_keyword_count = (
        count_suspicious_keywords(
            url
        )
    )

    shortened_url = (
        contains_shortener(
            hostname
        )
    )


    # -----------------------------------
    # Explicit port
    # -----------------------------------

    has_explicit_port = 0

    if parsed_url is not None:

        try:
            has_explicit_port = int(
                parsed_url.port
                is not None
            )

        except ValueError:
            has_explicit_port = 1
            parse_error = 1


    # -----------------------------------
    # Encoding
    # -----------------------------------

    percent_encoding_count = len(
        re.findall(
            r"%[0-9A-Fa-f]{2}",
            url,
        )
    )


    # -----------------------------------
    # Path indicators
    # -----------------------------------

    double_slash_in_path = int(
        "//" in path
    )

    suspicious_extension = (
        has_suspicious_extension(
            path
        )
    )


    # -----------------------------------
    # Final numerical feature dictionary
    # -----------------------------------

    features = {

        "url_length":
            url_length,

        "hostname_length":
            hostname_length,

        "path_length":
            path_length,

        "query_length":
            query_length,

        "dot_count":
            url.count("."),

        "hyphen_count":
            url.count("-"),

        "underscore_count":
            url.count("_"),

        "slash_count":
            url.count("/"),

        "question_mark_count":
            url.count("?"),

        "equal_count":
            url.count("="),

        "at_count":
            url.count("@"),

        "ampersand_count":
            url.count("&"),

        "digit_count":
            digit_count,

        "digit_ratio":
            round(
                digit_ratio,
                6,
            ),

        "letter_ratio":
            round(
                letter_ratio,
                6,
            ),

        "special_character_count":
            special_character_count,

        "special_character_ratio":
            round(
                special_character_ratio,
                6,
            ),

        "non_ascii_count":
            non_ascii_count,

        "non_ascii_ratio":
            round(
                non_ascii_ratio,
                6,
            ),

        "uses_https":
            uses_https,

        "has_ip_address":
            has_ip,

        "subdomain_count":
            subdomain_count,

        "suspicious_keyword_count":
            suspicious_keyword_count,

        "shortened_url":
            shortened_url,

        "has_explicit_port":
            has_explicit_port,

        "percent_encoding_count":
            percent_encoding_count,

        "double_slash_in_path":
            double_slash_in_path,

        "suspicious_extension":
            suspicious_extension,

        "url_parse_error":
            parse_error,

        "url_entropy":
            round(
                calculate_entropy(
                    url
                ),
                6,
            ),
    }

    return features