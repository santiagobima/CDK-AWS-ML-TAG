import re
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def format_duration(row):
    try:
        duration = pd.to_timedelta(row, unit='ms')
        days = duration.days
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days} days")
        if hours > 0:
            parts.append(f"{hours} hours")
        if minutes > 0:
            parts.append(f"{minutes} minutes")
        if seconds > 0:
            parts.append(f"{seconds} seconds")

        return ' '.join(parts[:2]) if parts else "0 seconds"

    except (ValueError, TypeError):
        return ''

def sanitize_string(s: str) -> str:
    """
    Sanitize the string by:
    - Removing special characters.
    - Replacing double white spaces with single ones.
    - Converting white spaces to underscores.
    - Converting to lowercase.
    """
    if pd.isna(s) or s is None:
        return "none"

    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)  # Remove special characters, keep alphanumerics and spaces.
    s = re.sub(r'\s+', ' ', s).strip()  # Replace multiple spaces with a single space.
    s = s.replace(' ', '_')  # Replace spaces with underscores.
    return s