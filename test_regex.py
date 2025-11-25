import re

lines = [
    "Basketball               725       0.716           0.986       29.67",
    "Biker                    142       0.255           0.254       57.51",
    "Overall                48689       0.524           0.617       22.41"
]

pattern = r'^(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'

for line in lines:
    match = re.match(pattern, line)
    if match:
        print(f"✅ Matched: {match.groups()}")
    else:
        print(f"❌ No match: {line}")
