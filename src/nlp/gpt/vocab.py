# 移除 torch 和 torch.nn 的导入
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional
# 移除 torchtext 相关导入
# from torchtext._torchtext import Vocab as VocabPybind
# from torchtext.utils import _log_class_usage

# --- 新的纯 Python Vocab 类 ---
class Vocab:
    r"""创建一个将 token 映射到索引的词汇表对象 (纯 Python 实现)。

    Args:
        itos (List[str]): 按索引顺序排列的 token 列表。
        unk_token (Optional[str]): 用于表示未知 token 的特殊符号。如果提供，
            查找未知 token 时将返回此 token 的索引。
    """
    def __init__(self, itos: List[str], unk_token: Optional[str] = None) -> None:
        super(Vocab, self).__init__()
        self._itos = itos
        self._stoi = {token: i for i, token in enumerate(itos)}
        self._unk_token = unk_token
        self._default_index = -1
        if unk_token is not None and unk_token in self._stoi:
            self._default_index = self._stoi[unk_token]

    def __len__(self) -> int:
        r"""
        Returns:
            词汇表的大小。
        """
        return len(self._itos)

    def __contains__(self, token: str) -> bool:
        r"""
        Args:
            token: 要检查是否存在的 token。

        Returns:
            token 是否在词汇表中。
        """
        return token in self._stoi

    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token: 用于查找对应索引的 token。

        Returns:
            与 token 关联的索引。如果 token 不存在且设置了默认索引，则返回默认索引。

        Raises:
            KeyError: 如果 token 不在词汇表中且未设置默认索引。
        """
        return self._stoi.get(token, self._default_index)

    def set_default_index(self, index: Optional[int]) -> None:
        r"""设置默认索引。当查询 OOV token 时将返回此索引。

        Args:
            index: 默认索引的值。如果为 None，则查询 OOV token 时会引发 KeyError。
                   通常设置为 unk_token 的索引。
        """
        # 在纯 Python 实现中，我们通常在初始化时通过 unk_token 设置，
        # 或者让用户直接通过 __getitem__ 的行为处理 OOV。
        # 为了保持接口一致性，我们添加这个方法，但主要依赖初始化时的 unk_token。
        # 如果需要更灵活的运行时更改，可以修改 __getitem__ 的逻辑。
        if index is not None:
             # 验证 index 是否有效 (可选)
             if not (0 <= index < len(self._itos)):
                 # 或者允许任意整数？当前实现只在 __getitem__ 中使用它
                 pass # 允许设置任意整数，但可能与 lookup_token 不一致
        self._default_index = index if index is not None else -1 # 使用 -1 表示未设置

    def get_default_index(self) -> Optional[int]:
        r"""
        Returns:
            如果设置了默认索引，则返回其值，否则返回 None (或内部表示 -1)。
        """
        return self._default_index if self._default_index != -1 else None

    def insert_token(self, token: str, index: int) -> None:
        r"""将 token 插入到指定索引处。不推荐在构建后频繁使用，
           因为它需要重建内部映射，效率较低。

        Args:
            token: 要插入的 token。
            index: 要插入到的索引位置。

        Raises:
            ValueError: 如果 token 已存在或索引越界。
        """
        if token in self._stoi:
            raise ValueError(f"Token '{token}' already exists in the vocabulary.")
        if not (0 <= index <= len(self._itos)): # 允许插入到末尾
             raise ValueError(f"Index {index} is out of range [0, {len(self._itos)}].")

        self._itos.insert(index, token)
        # 重建 stoi 映射
        self._stoi = {t: i for i, t in enumerate(self._itos)}
        # 如果默认索引受影响，需要更新 (如果它指向 unk_token)
        if self._unk_token is not None and self._unk_token in self._stoi:
             self._default_index = self._stoi[self._unk_token]
        elif self._default_index != -1 and self._default_index >= index:
             # 如果默认索引指向一个被后移的普通 token，也需要更新
             # 但这通常不符合默认索引的用法，所以我们主要关心 unk_token
             pass


    def append_token(self, token: str) -> None:
        r"""将 token 追加到词汇表末尾。

        Args:
            token: 要追加的 token。

        Raises:
            ValueError: 如果 token 已存在。
        """
        if token in self._stoi:
            raise ValueError(f"Token '{token}' already exists in the vocabulary.")
        self._itos.append(token)
        self._stoi[token] = len(self._itos) - 1

    def lookup_token(self, index: int) -> str:
        r"""
        Args:
            index: 要查找对应 token 的索引。

        Returns:
            与索引关联的 token。

        Raises:
            IndexError: 如果索引超出范围 [0, len(itos) - 1)。
        """
        if not (0 <= index < len(self._itos)):
            raise IndexError(f"Index ({index}) out of range [0, {len(self._itos)}).")
        return self._itos[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        r"""
        Args:
            indices: 用于查找对应 token 的索引列表。

        Returns:
            与索引列表关联的 token 列表。

        Raises:
            IndexError: 如果列表中的任何索引超出范围。
        """
        return [self.lookup_token(idx) for idx in indices]

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        r"""
        Args:
            tokens: 用于查找对应索引的 token 列表。

        Returns:
            与 token 列表关联的索引列表。 OOV token 会被映射到默认索引。
        """
        return [self.__getitem__(token) for token in tokens]

    def get_stoi(self) -> Dict[str, int]:
        r"""
        Returns:
            token 到索引的映射字典。
        """
        # 返回副本以防止外部修改
        return self._stoi.copy()

    def get_itos(self) -> List[str]:
        r"""
        Returns:
            索引到 token 的映射列表。
        """
        # 返回副本以防止外部修改
        return self._itos[:]

# --- 更新后的工厂函数 ---

def vocab(
    ordered_dict: OrderedDict, min_freq: int = 1, specials: Optional[List[str]] = None, special_first: bool = True
) -> Vocab:
    r"""创建 Vocab 对象的工厂方法 (使用纯 Python Vocab)。

    Args:
        ordered_dict: token 到频率的有序字典。
        min_freq: 包含 token 的最小频率。
        specials: 要添加的特殊符号列表。
        special_first: 是否将特殊符号放在开头。

    Returns:
        一个 Vocab 对象。
    """
    specials = specials or []
    # 从 ordered_dict 中移除 specials，以防它们也作为普通 token 出现
    # 注意：原始实现是在构建 tokens 列表 *之后* 才 pop，这里提前 pop 确保 specials 不会被计入普通 token
    for token in specials:
        ordered_dict.pop(token, None)

    # 按频率过滤 token
    filtered_tokens = [token for token, freq in ordered_dict.items() if freq >= min_freq]

    # 构建最终的 itos 列表
    itos = []
    if special_first:
        itos.extend(specials)
        itos.extend(filtered_tokens)
    else:
        itos.extend(filtered_tokens)
        itos.extend(specials)

    # 确定 unk_token (如果存在于 specials 中)
    # 假设 specials 列表中的第一个元素通常是 unk (如果存在的话)
    # 或者我们可以要求用户显式指定 unk_token
    unk_token = specials[0] if specials else None # 简单的假设

    # 使用新的 Vocab 类创建实例
    return Vocab(itos, unk_token=unk_token)


def build_vocab_from_iterator(
    iterator: Iterable,
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
    max_tokens: Optional[int] = None,
) -> Vocab:
    """
    从迭代器构建 Vocab (使用纯 Python Vocab)。

    Args:
        iterator: 用于构建 Vocab 的迭代器。必须产生 token 列表或迭代器。
        min_freq: 包含 token 的最小频率。
        specials: 要添加的特殊符号列表。
        special_first: 是否将特殊符号放在开头。
        max_tokens: 如果提供，则从 `max_tokens - len(specials)` 个最频繁的 token 创建词汇表。

    Returns:
        一个 Vocab 对象。
    """
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = specials or []

    # 首先按频率降序排序，然后按字典序排序
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_tokens is None:
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
    else:
        num_special = len(specials)
        if max_tokens <= num_special:
             # 如果 max_tokens 小于等于 specials 数量，只保留 specials
             # 或者可以抛出错误，因为这可能不是预期行为
             print(f"Warning: max_tokens ({max_tokens}) <= len(specials) ({num_special}). "
                   f"Vocabulary will only contain special tokens.")
             ordered_dict = OrderedDict() # 没有普通 token
        else:
             # 否则，取 top N 个普通 token
             ordered_dict = OrderedDict(sorted_by_freq_tuples[:max_tokens - num_special])


    # 调用更新后的 vocab 工厂函数
    word_vocab = vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=special_first)

    # 检查是否需要设置默认索引 (例如，如果 unk_token 不在 specials 的首位)
    # 假设 unk_token 是 specials[0] (如果 specials 非空)
    unk_token = specials[0] if specials else None
    if unk_token is not None:
        if unk_token in word_vocab:
            word_vocab.set_default_index(word_vocab[unk_token])
        else:
            # 如果 unk_token 因为 min_freq 或 max_tokens 被过滤掉了，
            # 并且它在 specials 列表中，这通常不应该发生，
            # 但如果发生了，默认索引将不会被设置，或者需要不同的逻辑。
            # 当前 vocab 函数会确保 specials 总是被包含。
            pass
    # else: # 如果没有 unk_token，默认索引保持为 -1 (或 None)

    return word_vocab

# --- 移除旧的 Vocab 类和 __prepare_scriptable__ ---
# (旧的 Vocab 类定义已被上面的新类替换)