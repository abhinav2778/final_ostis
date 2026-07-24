# ─────────────────────────────────────────────────────────────────
# OSTIS — Shared entity lists for NER
#
# Single source of truth so extract_entities.py, finetune_ner.py, and
# ner_combiner.py all agree on the same known entities and blocklist.
# Sorted by length descending on import so multi-word phrases
# ("lazarus group") are matched before their substrings ("lazarus").
# ─────────────────────────────────────────────────────────────────

MALWARE_LIST = [
    "phantomraven", "openclaw", "log4j", "log4shell",
    "sparrowdoor", "crowdoor", "lockbit", "blackcat", "alphv", "conti",
    "revil", "ryuk", "darkside", "blackmatter", "hive", "clop", "maze",
    "ragnar", "ransomhouse", "akira", "play", "royal", "cuba", "nokoyawa",
    "blackbyte", "lorenz", "vice society", "snatch", "wannacry",
    "notpetya", "bad rabbit", "petya", "locky", "cerber", "gandcrab",
    "emotet", "trickbot", "qakbot", "dridex", "icedid", "bazarloader",
    "cobalt strike", "metasploit", "mimikatz", "lazagne",
    "remcos", "asyncrat", "njrat", "darkcomet", "quasar",
    "whispergate", "hermetic wiper", "industroyer", "triton",
    "industroyer2", "crashoverride", "pegasus", "predator",
    "redline", "vidar", "raccoon", "formbook", "agent tesla",
    "mirai", "3am", "cl0p", "blacksuit", "medusa",
]

THREAT_ACTOR_LIST = [
    "lazarus group", "lazarus", "apt28", "fancy bear", "sednit",
    "apt29", "cozy bear", "midnight blizzard", "apt41", "winnti",
    "apt10", "stone panda", "apt1", "sandworm", "kimsuky",
    "scattered spider", "lapsus$", "lapsus", "0ktapus",
    "charming kitten", "apt35", "volt typhoon", "salt typhoon",
    "cl0p gang", "inc ransomware", "muddled libra",
    "shinyhunters", "kimwolf", "honeymyte", "famoussparrow",
    "china nexus", "uat-5918", "uat-9244",
]

# Legitimate tech companies / products that a general NER model tends to
# flag as ORG/MISC. These are NOT threat entities — filtered out of any
# "unknown candidate" output.
VENDOR_BLOCKLIST = {
    "microsoft", "windows", "google", "apple", "linux", "android",
    "cisco", "talos", "cisco talos", "dell", "intel", "amd", "oracle",
    "amazon", "aws", "azure", "github", "gitlab", "adobe", "vmware",
    "fortinet", "paloalto", "palo alto", "palo alto networks",
    "crowdstrike", "sentinelone", "sophos", "symantec", "mcafee",
    "norton", "kaspersky", "bitdefender", "office", "outlook", "teams",
    "sharepoint", "onedrive", "chrome", "firefox", "safari", "iphone",
    "ipad", "macos", "ios", "python", "java", "php", "javascript",
    "powershell", "bash", "united states", "russia", "china", "iran",
    "north korea", "europe", "european", "federal", "national",
    "government", "samsung", "entra", "claroty", "dragos",
    "malwarebytes", "threatpost", "cisa", "starbucks", "loblaw",
    "canadian", "covid", "cissp", "sase", "siem", "genai", "mssps",
    "msps", "dfir", "coruna", "stryker", "team82", "raas", "cisos",
    "microsoft exchange", "microsoft windows",
    "microsoft windows operating system", "swifdoo", "chrome zero",
    "clamav", "foxit", "bugsplat",
}

# Sort by phrase length descending so longer matches win first.
MALWARE_LIST.sort(key=len, reverse=True)
THREAT_ACTOR_LIST.sort(key=len, reverse=True)

# Some malware/ransomware group names are also common English words
# ("play", "royal", "hive", "maze", "cuba", "medusa", "snatch"). Matching
# these on the word alone produces heavy false positives ("plug-and-play",
# "cisco booth play", "continues to play a role"). For these specific
# entries, require a nearby context word in the same sentence before
# counting it as a real match.
AMBIGUOUS_TERM_CONTEXT = {
    "play": ["ransomware", "ransom", "gang", "extortion", "leak site"],
    "royal": ["ransomware", "ransom", "gang", "extortion", "leak site"],
    "hive": ["ransomware", "ransom", "gang", "extortion", "leak site"],
    "maze": ["ransomware", "ransom", "gang", "extortion", "leak site"],
    "cuba": ["ransomware", "ransom", "gang", "extortion", "leak site"],
    "medusa": ["ransomware", "ransom", "gang", "extortion", "leak site"],
    "snatch": ["ransomware", "ransom", "gang", "extortion", "leak site"],
}
