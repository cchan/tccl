import math
from functools import lru_cache


def build_tree(N):
    """
    See docs in build_double_tree for more details...
    """

    # lol well since we're only testing with 8 GPUs...
    assert (N & (N - 1)) == 0, "len(ranks) is not a power of 2!"

    # Build the tree from the bottom (leaf nodes) up
    height = int(math.log2(N))
    tree = []
    prev_row = None
    for h in range(height):
        if prev_row is None:
            # First step -- bottom-most (leaf) row is all the odd ranks
            row = [i for i in range(N) if i % 2 != 0]
        else:
            # Value for each rank in the non-leaf row is the middle between the 2 children below
            row = []
            for i in range(0, len(prev_row), 2):
                row.append((prev_row[i] + prev_row[i+1]) // 2)
        tree = row + tree
        prev_row = row

    # Finally add the subroot (since we have an extra node from being a power of 2 -- a perfect binary tree has 2**N-1 nodes)
    tree = [0] + tree
    return tree
    

@lru_cache
def build_double_tree(N):
    """
    Build a double binary tree in the pattern in https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4

    In each binary tree, a rank will wait to receive messages from its children (if any),
    and then do the reduction with both of the children and it's own value,
    and then send the reduced value to the parent.

    Once the root has finished the above process, it will have the fully reduced value.
    It the needs to broadcast that value down to its children (and the children do the same after receiving the value from their parent and so on)
    until the leaf nodes all have the fully reduced value.

    We can instead build 2 binary trees in such a way that a given rank is a leaf in the first one and a non-leaf in the other one.
    Then, in the tree where a given rank is a leaf it will be sending 1 unit of data (to its parent), while in the other tree where that rank is a non-leaf
    it will be receiving 2 units of data (from its children) and sending 1 unit of data (to its parent).
    As a result each rank is sending and receiving the same amount of data!

    Note: If there is only one child (e.g. the subroot in the trees), then only the left child is set while the right child is -1.

    E.g. if N is 8, the trees will be constructed like:
               0
             /
           4
        /     \\
        2      6
       / \\   / \\
      1   3   5  7

         AND

               7
             /
           3
        /     \\
        1      5
       / \\   / \\
      0   2   4  6

    tree_a will be [0, 4, 2, 6, 1, 3, 5, 7]
    tree_b will be [7, 3, 1, 5, 0, 2, 4, 6]

    To find the neighbors of a rank, find the index of that rank. Then,
       The parent is floor(index / 2)
       The left child is index * 2
       The right child is index * 2 + 1
       Out of bounds means there is no child.
    ... except for the special case of the subroot (0 and 7 in the above example), which are hardcoded.
    """
    a = build_tree(N)
    # To generate the second complementary tree, simply subtract 1 from ever element in the first tree.
    # Note the subroot (0) will go around to the largest value (instead of going to -1)
    b = build_tree(N)
    b = list(map(lambda e: e - 1, b))
    assert b[0] == -1
    b[0] = N - 1

    return a, b

def get_parents_and_children(N, rank):
    """
    Note: If there is only one child (e.g. the subroot in the trees), then only the left child is set while the right child is -1.
    In all other cases, either:
    - both children are -1
    - both children are not -1

    E.g. if N is 8, the trees will be constructed like:
               0
             /
           4
        /     \\
        2      6
       / \\   / \\
      1   3   5  7
         AND
               7
             /
           3
        /     \\
        1      5
       / \\   / \\
      0   2   4  6

    tree_a will be [0, 4, 2, 6, 1, 3, 5, 7]
    tree_b will be [7, 3, 1, 5, 0, 2, 4, 6]

    To find the neighbors of a rank, find the index of that rank. Then,
       The parent is floor(index / 2)
       The left child is index * 2
       The right child is index * 2 + 1
       Out of bounds means there is no child.
    Except for the special case of the subroot (0 and 7 in the above example), which are hardcoded.
    """

    tree_a, tree_b = build_double_tree(N)

    def get_tree_parent_and_children(tree, rank):
        rank_index = tree.index(rank)
        # Special case for subroot -- only 1 child
        if rank_index == 0:
            parent = -1
            left_child = tree[rank_index + 1]
            right_child = -1
        else:
            parent_idx = rank_index // 2
            left_child_idx = rank_index * 2
            right_child_idx = left_child_idx + 1

            parent = tree[parent_idx]
            left_child = tree[left_child_idx] if left_child_idx < len(tree) else -1
            right_child = tree[right_child_idx] if right_child_idx < len(tree) else -1

        return parent, left_child, right_child

    return *get_tree_parent_and_children(tree_a, rank), *get_tree_parent_and_children(tree_b, rank)


if __name__ == "__main__":
    WORLD_SIZE = 8
    # Reference: ASCII art above
    tree_a, tree_b = build_double_tree(WORLD_SIZE)
    assert tree_a == [0, 4, 2, 6, 1, 3, 5, 7]
    assert len(set(tree_a)) == len(tree_a)
    assert tree_b == [7, 3, 1, 5, 0, 2, 4, 6]
    assert len(set(tree_b)) == len(tree_b)
    #                                                  tree_a          tree_b
    assert get_parents_and_children(WORLD_SIZE, 0) == (-1, 4, -1,    1, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 1) == (2, -1, -1,    3, 0, 2)
    assert get_parents_and_children(WORLD_SIZE, 2) == (4, 1, 3,      1, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 3) == (2, -1, -1,    7, 1, 5)
    assert get_parents_and_children(WORLD_SIZE, 4) == (0, 2, 6,      5, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 5) == (6, -1, -1,    3, 4, 6)
    assert get_parents_and_children(WORLD_SIZE, 6) == (4, 5, 7,      5, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 7) == (6, -1, -1,    -1, 3, -1)

    # Reference:
    """
        0
       /
      2
     / \\
    1   3

    AND 
        3
       /
      1
     / \\
    0   2
    """
    WORLD_SIZE = 4
    tree_a, tree_b = build_double_tree(WORLD_SIZE)
    assert tree_a == [0, 2, 1, 3]
    assert len(set(tree_a)) == len(tree_a)
    assert tree_b == [3, 1, 0, 2]
    assert len(set(tree_b)) == len(tree_b)
    #                                                  tree_a          tree_b
    assert get_parents_and_children(WORLD_SIZE, 0) == (-1, 2, -1,    1, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 1) == (2, -1, -1,    3, 0, 2)
    assert get_parents_and_children(WORLD_SIZE, 2) == (0, 1, 3,      1, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 3) == (2, -1, -1,    -1, 1, -1)
 

    WORLD_SIZE = 16
    tree_a, tree_b = build_double_tree(WORLD_SIZE)
    assert tree_a == [0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]
    assert len(set(tree_a)) == len(tree_a)
    assert tree_b == [15, 7, 3, 11, 1, 5, 9, 13, 0, 2, 4, 6, 8, 10, 12, 14]
    assert len(set(tree_b)) == len(tree_b)

    # Reference: Image in https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/
    WORLD_SIZE = 32
    tree_a, tree_b = build_double_tree(WORLD_SIZE)
    assert len(set(tree_a)) == len(tree_a)
    assert tree_a == [0, 16, 8, 24, 4, 12, 20, 28, 2, 6, 10, 14, 18, 22, 26, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    assert len(set(tree_b)) == len(tree_b)
    assert tree_b == [31, 15, 7, 23, 3, 11, 19, 27, 1, 5, 9, 13, 17, 21, 25, 29, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    #                                                  tree_a            tree_b
    assert get_parents_and_children(WORLD_SIZE, 0) == (-1, 16, -1,    1, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 1) == (2, -1, -1,     3, 0, 2)
    assert get_parents_and_children(WORLD_SIZE, 2) == (4, 1, 3,       1, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 3) == (2, -1, -1,     7, 1, 5)
    assert get_parents_and_children(WORLD_SIZE, 4) == (8, 2, 6,       5, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 5) == (6, -1, -1,     3, 4, 6)
    assert get_parents_and_children(WORLD_SIZE, 6) == (4, 5, 7,       5, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 7) == (6, -1, -1,     15, 3, 11)
    assert get_parents_and_children(WORLD_SIZE, 8) == (16, 4, 12,     9, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 9) == (10, -1, -1,    11, 8, 10)
    assert get_parents_and_children(WORLD_SIZE, 10) == (12, 9, 11,    9, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 11) == (10, -1, -1,   7, 9, 13)
    assert get_parents_and_children(WORLD_SIZE, 12) == (8, 10, 14,    13, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 13) == (14, -1, -1,   11, 12, 14)
    assert get_parents_and_children(WORLD_SIZE, 14) == (12, 13, 15,   13, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 15) == (14, -1, -1,   31, 7, 23)
    assert get_parents_and_children(WORLD_SIZE, 16) == (0, 8, 24,     17, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 17) == (18, -1, -1,   19, 16, 18)
    assert get_parents_and_children(WORLD_SIZE, 18) == (20, 17, 19,   17, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 19) == (18, -1, -1,   23, 17, 21)
    assert get_parents_and_children(WORLD_SIZE, 20) == (24, 18, 22,   21, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 21) == (22, -1, -1,   19, 20, 22)
    assert get_parents_and_children(WORLD_SIZE, 22) == (20, 21, 23,   21, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 23) == (22, -1, -1,   15, 19, 27)
    assert get_parents_and_children(WORLD_SIZE, 24) == (16, 20, 28,   25, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 25) == (26, -1, -1,   27, 24, 26)
    assert get_parents_and_children(WORLD_SIZE, 26) == (28, 25, 27,   25, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 27) == (26, -1, -1,   23, 25, 29)
    assert get_parents_and_children(WORLD_SIZE, 28) == (24, 26, 30,   29, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 29) == (30, -1, -1,   27, 28, 30)
    assert get_parents_and_children(WORLD_SIZE, 30) == (28, 29, 31,   29, -1, -1)
    assert get_parents_and_children(WORLD_SIZE, 31) == (30, -1, -1,   -1, 15, -1)

    print("All passed!")
