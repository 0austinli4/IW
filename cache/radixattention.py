from __future__ import annotations
import heapq
import time
from typing import Optional, List, Callable

class EvictionPolicy:
    def evict(self, radix_cache: RadixCache, num_tokens: int, evict_callback: Callable):
        raise NotImplementedError("Eviction policy must implement evict()")

### Parent Class for LRU, FIFO, LFU
class EvictLeafComparison(EvictionPolicy):
    def evict(self, radix_cache: RadixCache, num_tokens: int, evict_callback: Callable):
        leaves = radix_cache._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == radix_cache.root:
                break

            evict_callback(x.value)
            num_evicted += len(x.value)
            radix_cache._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

### FIFO REINSERTION
class FifoReinsertion(EvictionPolicy):
    def evict(self, radix_cache: RadixCache, num_tokens: int, evict_callback: Callable):
        leaves = radix_cache._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            ### FIFO REINSERTION FOR ACCESS
            if x.accessed == 1:
                x.accessed = 0
                # insert to back of the queue
                x.creation_time = time.time()
                heapq.heappush(leaves, x)
                continue

            if x == radix_cache.root:
                break

            evict_callback(x.value)
            num_evicted += len(x.value)
            radix_cache._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

class TreeNode:
    def __init__(self, eviction_policy_name: str, eviction_policy: EvictionPolicy, key: str = "", value: Optional[str] = None):
        self.key = key              # The segment for this node.
        self.value = value          # Stored value (only at leaves).
        self.children: dict[str, TreeNode] = {}
        self.creation_time = time.time()
        self.last_access_time = time.time()
        self.eviction_policy = eviction_policy
        self.eviction_policy_name = eviction_policy_name
        self.hit_count = 0
        # FOR FIFO REINSERTION - 0 if not accessed 1 if did
        self.accessed = 0 

    def __lt__(self, other: TreeNode):
        if self.eviction_policy_name == "lru":
            return self.last_access_time < other.last_access_time
        elif self.eviction_policy_name == "fifo" or self.eviction_policy_name == "fifo_reinsertion":
            return self.creation_time < other.creation_time
        elif self.eviction_policy_name == "lfu":
            return self.hit_count < other.hit_count
        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy_name}")

def _key_match(key0: str, key1: str) -> int:
    """Return the number of matching characters between two strings."""
    i = 0
    for c0, c1 in zip(key0, key1):
        if c0 != c1:
            break
        i += 1
    return i

class RadixCache:
    def __init__(self, eviction_policy: str = "lru"):
        self.root = TreeNode()
        self.eviction_policy = eviction_policy

    def insert(self, key: str, value: Optional[str] = None):
        """
        Insert a key/value pair into the radix tree.
        If value is None, the key itself is stored as the value.
        """
        if value is None:
            value = key
        node = self.root
        node.last_access_time = time.time()
        while key:
            first_char = key[0]
            if first_char in node.children:
                child = node.children[first_char]
                child.last_access_time = time.time()
                match_len = _key_match(child.key, key)
                # If the match is partial, split the child node.
                if match_len < len(child.key):
                    self._split_node(child, match_len)
                # If key still has remaining characters, continue traversing.
                if match_len < len(key):
                    key = key[match_len:]
                    node = child
                    continue
                else:
                    # Exact match: update the value.
                    child.value = value
                    return
            else:
                # Create a new child node for the remaining key.
                new_node = TreeNode(key, value)
                node.children[first_char] = new_node
                return

    def _split_node(self, node: TreeNode, split_pos: int):
        """Split a node into two at split_pos."""
        old_key = node.key
        old_value = node.value
        old_children = node.children

        # Create a new node for the remainder of the key.
        remainder = TreeNode(old_key[split_pos:], old_value)
        remainder.children = old_children
        remainder.last_access_time = node.last_access_time

        # Update the current node to hold only the common prefix.
        node.key = old_key[:split_pos]
        node.value = None  # Intermediate nodes hold no value.
        node.children = {remainder.key[0]: remainder}

    def get(self, key: str) -> Optional[str]:
        """
        Lookup a key in the radix tree.
        Returns the stored value if the key exactly matches, else None.
        """
        node = self.root
        node.last_access_time = time.time()
        while key:
            first_char = key[0]
            if first_char in node.children:
                child = node.children[first_char]
                child.last_access_time = time.time()
                ### FOR FIFO REINSERTION LAST ACCESSED
                child.accessed = 1
                if key.startswith(child.key):
                    key = key[len(child.key):]
                    child.hit_count += 1
                    node = child
                else:
                    return None
            else:
                return None
        return node.value

    def evict(self, num_tokens: int, evict_callback: Callable):
        self.eviction_policy.evict(self, num_tokens, evict_callback)

    def _delete_leaf(self, node: TreeNode):
        """Delete a leaf node from the tree by finding its parent."""
        parent = self._find_parent(self.root, node)
        if parent:
            for k, child in list(parent.children.items()):
                if child == node:
                    del parent.children[k]
                    break

    def _find_parent(self, current: TreeNode, target: TreeNode) -> Optional[TreeNode]:
        """Recursively find the parent of the target node."""
        for child in current.children.values():
            if child == target:
                return current
            res = self._find_parent(child, target)
            if res:
                return res
        return None

    def _collect_leaves(self) -> List[TreeNode]:
        """Collect all leaf nodes in the tree."""
        leaves = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not node.children:
                leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def pretty_print(self):
        """Print the tree in a human-readable format."""
        self._print_helper(self.root, 0)

    def _print_helper(self, node: TreeNode, indent: int):
        print(" " * indent + f"Key: {node.key!r}, Value: {node.value!r}")
        for child in node.children.values():
            self._print_helper(child, indent + 2)
        
    # def get_flops_efficiency(self, l, d, n, num_mamba_layers, num_attn_layers, num_mlp_layers):
    #     total_flops_saved = num_mamba_layers * get_mamba1_flops(l, d, n) + \
    #         num_attn_layers * get_attn_flops(l, d) + \
    #         num_mlp_layers * get_mlp_flops(l, d)
    #     total_state_size = get_model_state_size(l, d, n, num_mamba_layers=num_mamba_layers, num_attn_layers=num_attn_layers)
    #     return total_flops_saved / total_state_size

if __name__ == "__main__":
    cache = RadixCache()

    # Test insertions
    cache.insert("Hello")
    cache.insert("Hello")         # Duplicate key; value is updated.
    cache.insert("Hello_L.A.!")
    
    print("Initial cache:")
    cache.pretty_print()

    # Test get operations
    print("\nGet operations:")
    print("Get 'Hello':", cache.get("Hello"))
    print("Get 'Hello_L.A.!':", cache.get("Hello_L.A.!"))
    print("Get 'NotThere':", cache.get("NotThere"))

    # Test eviction (evicting the least recently used leaf)
    print("\nEviction:")
    def evict_callback(val):
        print("Evicted:", val)
    cache.evict(evict_callback)
    
    print("\nCache after eviction:")
    cache.pretty_print()
