import random
from .models import MemoryItem

# this is a dummy classifier - randomly assigns tier
def classifier(item: MemoryItem) -> None:
    random_tier = random.choice(["SCRATCH", "SESSION", "LONGTERM"])
    item.tier = random_tier
    item.trust_score = random.random()
