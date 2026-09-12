import ipaddress
import math
import re
from collections import Counter
from urllib.parse import parse_qsl, unquote, urlsplit


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


SHORTENED_URL_DOMAINS = {
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


SCHEME_PATTERN = re.compile(
    r"^[a-zA-Z][a-zA-Z0-9+.-]*://"
)


def safe_parse_url(url: str):
    """
    Parse a URL safely.

    URLs without a scheme are temporarily given http:// only for parsing.
    The temporary scheme is never used in lexical feature calculation.
    """

    parse_error = 0

    value = url.strip()

    if value.startswith("//"):
        parse_value = "http:" + value
    elif SCHEME_PATTERN.match(value):
        parse_value = value
    else:
        parse_value = "http://" + value

    try:
        parsed = urlsplit(parse_value)

        # Access hostname here because malformed brackets can
        # raise ValueError only when hostname is requested.
        _ = parsed.hostname

        return parsed, parse_error

    except (ValueError, UnicodeError):
        parse_error = 1

    # Retry malformed brackets in a safer representation.
    repaired = (
        parse_value
        .replace("[", "%5B")
        .replace("]", "%5D")
    )

    try:
        parsed = urlsplit(repaired)
        _ = parsed.hostname

        return parsed, parse_error

    except (ValueError, UnicodeError):
        return None, parse_error


def get_hostname(parsed) -> str:
    if parsed is None:
        return ""

    try:
        hostname = parsed.hostname or ""
    except (ValueError, UnicodeError):
        return ""

    hostname = hostname.strip().lower().rstrip(".")

    return hostname


def normalize_hostname(hostname: str) -> str:
    """
    Remove the common www. prefix so:
    google.com
    www.google.com
    are treated similarly by lexical features.
    """

    if hostname.startswith("www."):
        return hostname[4:]

    return hostname


def normalize_path(path: str) -> str:
    """
    Treat a root slash as an empty path.

    example.com
    example.com/

    should not become meaningfully different samples.
    """

    if not path or path == "/":
        return ""

    return path


def is_ip_address(hostname: str) -> int:
    if not hostname:
        return 0

    candidate = hostname

    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]

    try:
        ipaddress.ip_address(candidate)
        return 1
    except ValueError:
        return 0


def count_subdomains(hostname: str) -> int:
    """
    Approximate subdomain count.

    Leading www is already removed.
    IP addresses return zero.
    """

    if not hostname:
        return 0

    if is_ip_address(hostname):
        return 0

    parts = [
        part
        for part in hostname.split(".")
        if part
    ]

    if len(parts) <= 2:
        return 0

    return len(parts) - 2


def get_tld_length(hostname: str) -> int:
    if not hostname:
        return 0

    if is_ip_address(hostname):
        return 0

    parts = [
        part
        for part in hostname.split(".")
        if part
    ]

    if len(parts) < 2:
        return 0

    return len(parts[-1])


def is_shortened_url(hostname: str) -> int:
    if not hostname:
        return 0

    for domain in SHORTENED_URL_DOMAINS:
        if hostname == domain:
            return 1

        if hostname.endswith("." + domain):
            return 1

    return 0


def has_suspicious_extension(path: str) -> int:
    if not path:
        return 0

    cleaned_path = unquote(path).lower().rstrip("/")

    return int(
        any(
            cleaned_path.endswith(extension)
            for extension in SUSPICIOUS_EXTENSIONS
        )
    )


def count_suspicious_keywords(text: str) -> int:
    decoded = unquote(text).lower()

    return sum(
        1
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in decoded
    )


def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0

    length = len(text)
    counts = Counter(text)

    entropy = 0.0

    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def calculate_ratio(
    count: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return count / total


def get_explicit_port(parsed) -> int:
    if parsed is None:
        return 0

    try:
        return int(parsed.port is not None)
    except ValueError:
        return 0


def get_query_parameter_count(query: str) -> int:
    if not query:
        return 0

    try:
        return len(
            parse_qsl(
                query,
                keep_blank_values=True,
            )
        )
    except ValueError:
        return len(
            [
                part
                for part in query.split("&")
                if part
            ]
        )


def get_path_depth(path: str) -> int:
    if not path:
        return 0

    return len(
        [
            segment
            for segment in path.split("/")
            if segment
        ]
    )


def extract_url_features(url: str) -> dict:
    if not isinstance(url, str):
        raise TypeError(
            "URL must be provided as a string."
        )

    raw_url = url.strip()

    if not raw_url:
        raise ValueError(
            "URL cannot be empty."
        )

    parsed, parse_error = safe_parse_url(
        raw_url
    )

    hostname = get_hostname(parsed)
    hostname = normalize_hostname(hostname)

    if parsed is not None:
        path = normalize_path(
            parsed.path or ""
        )

        query = parsed.query or ""

    else:
        path = ""
        query = ""

    # ---------------------------------------------------------
    # Scheme-neutral representation
    # ---------------------------------------------------------
    #
    # These should produce nearly identical lexical features:
    #
    # google.com
    # http://google.com
    # https://www.google.com
    #
    # Protocol formatting therefore cannot dominate the model.

    lexical_url = hostname + path

    if query:
        lexical_url += "?" + query

    lexical_length = len(lexical_url)

    # ---------------------------------------------------------
    # Character statistics
    # ---------------------------------------------------------

    digit_count = sum(
        character.isdigit()
        for character in lexical_url
    )

    letter_count = sum(
        character.isalpha()
        for character in lexical_url
    )

    special_character_count = sum(
        not character.isalnum()
        for character in lexical_url
    )

    non_ascii_count = sum(
        ord(character) > 127
        for character in lexical_url
    )

    hostname_digit_count = sum(
        character.isdigit()
        for character in hostname
    )

    hostname_special_count = sum(
        not character.isalnum()
        and character != "."
        for character in hostname
    )

    hostname_length = len(hostname)

    keyword_text = (
        hostname
        + path
        + "?"
        + query
    )

    percent_encoding_text = (
        path
        + "?"
        + query
    )

    # ---------------------------------------------------------
    # Final feature dictionary
    # ---------------------------------------------------------

    return {
        "url_length": lexical_length,

        "hostname_length": hostname_length,

        "path_length": len(path),

        "query_length": len(query),

        "dot_count": lexical_url.count("."),

        "hyphen_count": lexical_url.count("-"),

        "underscore_count": lexical_url.count("_"),

        "slash_count": path.count("/"),

        "question_mark_count": (
            1 if query else 0
        ),

        "equal_count": query.count("="),

        # @ is scheme-neutral, so preserve it from raw URL.
        "at_count": raw_url.count("@"),

        "ampersand_count": query.count("&"),

        "digit_count": digit_count,

        "digit_ratio": calculate_ratio(
            digit_count,
            lexical_length,
        ),

        "letter_ratio": calculate_ratio(
            letter_count,
            lexical_length,
        ),

        "special_character_count":
            special_character_count,

        "special_character_ratio":
            calculate_ratio(
                special_character_count,
                lexical_length,
            ),

        "non_ascii_count": non_ascii_count,

        "non_ascii_ratio":
            calculate_ratio(
                non_ascii_count,
                lexical_length,
            ),

        "has_ip_address":
            is_ip_address(hostname),

        "subdomain_count":
            count_subdomains(hostname),

        "suspicious_keyword_count":
            count_suspicious_keywords(
                keyword_text
            ),

        "shortened_url":
            is_shortened_url(hostname),

        "has_explicit_port":
            get_explicit_port(parsed),

        "percent_encoding_count":
            len(
                re.findall(
                    r"%[0-9A-Fa-f]{2}",
                    percent_encoding_text,
                )
            ),

        "double_slash_in_path":
            int("//" in path),

        "suspicious_extension":
            has_suspicious_extension(path),

        "url_parse_error":
            parse_error,

        "url_entropy":
            calculate_entropy(
                lexical_url
            ),

        # -----------------------------------------------------
        # New v2 structural features
        # -----------------------------------------------------

        "hostname_digit_ratio":
            calculate_ratio(
                hostname_digit_count,
                hostname_length,
            ),

        "hostname_special_ratio":
            calculate_ratio(
                hostname_special_count,
                hostname_length,
            ),

        "path_depth":
            get_path_depth(path),

        "query_parameter_count":
            get_query_parameter_count(
                query
            ),

        "tld_length":
            get_tld_length(hostname),
    }