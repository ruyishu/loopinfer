"""Bounded immutable full-page prefix reuse within a loaded engine."""

from collections import OrderedDict
from hashlib import sha256
from struct import pack


class PrefixCache:
    def __init__(self, pool, max_pages):
        self.pool = pool
        self.max_pages = max_pages
        self.entries = OrderedDict()
        self.owner = -1
        self.hit_tokens = self.evictions = 0

    def keys(self, tokens, depth, namespace):
        if not self.max_pages:
            return ()
        # The cache is private to one set of weights, dtype and execution settings.
        digest = sha256(pack(">I", depth) + namespace.encode()).digest()
        size = self.pool.page_size
        keys = []
        # Leave at least one real prompt token to compute the first logits.
        for offset in range(0, len(tokens) - size, size):
            digest = sha256(digest + pack(f">{size}I", *tokens[offset:offset + size])).digest()
            keys.append(digest)
        return tuple(keys)

    def attach(self, request):
        for key in request.prefix_keys:
            if key not in self.entries:
                break
            page, _ = self.entries[key]
            self.pool.retain(request.handle.id, page)
            request.pages.append(page)
            self.entries.move_to_end(key)
        request.cursor = len(request.pages) * self.pool.page_size
        self.hit_tokens += request.cursor

    def publish(self, request, previous, current):
        if not request.prefix_keys:
            return
        size = self.pool.page_size
        end = min(len(request.prefix_keys), current // size)
        for index in range(previous // size, end):
            key = request.prefix_keys[index]
            if key in self.entries:
                continue
            while len(self.entries) >= self.max_pages:
                self.evict()
            page = request.pages[index]
            self.pool.retain(self.owner, page)
            self.entries[key] = (page, self.owner)
            self.owner -= 1

    def evict(self):
        if not self.entries:
            return False
        _, (page, owner) = self.entries.popitem(last=False)
        self.pool.drop(owner, page)
        self.evictions += 1
        return True
