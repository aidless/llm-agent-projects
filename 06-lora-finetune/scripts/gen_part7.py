"""Part 7: 60+ entries (safer approach)."""
import json, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "sample", "train.jsonl"))

# Simpler entries without complex string escaping
entries = [
    ("Python 中 lambda 与 def 性能差异？",
     "lambda 与 def 性能差异微乎其微，lambda 创建稍快但调用相同。\n两者产生的字节码几乎相同，调用速度也基本相同。选择哪个不应基于性能，而应基于可读性。lambda 适合简单一次性函数，def 适合复杂逻辑且需要文档的函数。"),
    ("Python 字符串拼接的 4 种方式性能对比？",
     "Python 字符串拼接的 4 种方式：加号运算符 O(n²) 最慢，join() O(n) 最快，f-string 与加号相同，StringIO 适合流式。大量字符串拼接应使用 join() 或 StringIO。性能测试显示 join 比加号快 100 倍以上。"),
    ("Python 中如何实现一个单例模式的多种方法？",
     "单例模式实现方法：1) 模块级实例（最简单）2) __new__ 方法 3) 装饰器 4) 元类 5) functools.cache。生产建议优先模块单例或依赖注入，避免单例。"),
    ("Python 中 list 排序 sorted() 怎么用？",
     "sorted() 返回新列表不修改原列表。常用参数：key 自定义排序键，reverse 反向排序。key 可以是函数或 itemgetter 等。多级排序用元组 key 如 lambda x: (x.age, -x.score)。Python sorted 是稳定排序。list.sort() 是原地版本。"),
    ("Python 中 dict 怎么实现？性能特点是什么？",
     "dict 基于哈希表实现，平均 O(1) 操作。Python 3.7+ 保证插入顺序。dict 视图（keys/values/items）动态反映 dict 变化。dict 推导式创建字典。Python 3.9+ 有 | 合并操作符。dict 是最常用的数据结构。"),
    ("Python 中 list 的内部实现？和 array.array 的区别？",
     "list 是动态数组（指针数组），array.array 是紧凑数组（连续内存）。list 灵活但占内存多，array 紧凑但只能存单一类型。array.array 适合数值密集场景，比 list 快约 3 倍。"),
    ("Python 的字符串方法 count、find、index 有什么区别？",
     "str.find 找不到返回 -1，str.index 找不到抛 ValueError，str.count 返回出现次数。str.startswith/endswith 支持元组前缀/后缀。str.split 限制次数。str.join 拼接序列。str.strip 去除空白或指定字符。"),
    ("Python 中如何处理二进制数据（bytes、bytearray、memoryview）？",
     "bytes 不可变，bytearray 可变，memoryview 零拷贝视图。整数转换用 to_bytes/from_bytes。编码用 encode/decode。Base64 用于文本传输。struct 用于二进制打包。memoryview 适合大数据切片零拷贝。"),
    ("Python 中如何处理大文件（GB 级）？",
     "大文件用流式处理：for line in f（按行）、按块读取（f.read(CHUNK)）、Pandas chunksize、ijson 流式 JSON、Dask/Polars 惰性求值。永远不要用 read() 读大文件。"),
    ("Python 3.11 的 tomllib 模块是什么？怎么用？",
     "tomllib（PEP 680）是 Python 3.11 引入的 TOML 解析标准库。tomllib.load(f) 读取文件（bytes 模式），tomllib.loads(s) 解析字符串。仅支持读取，写入用第三方 tomli_w。3.13 重写为 Rust 实现性能提升 2-3 倍。"),
    ("Python typing 模块：ParamSpec 怎么用？（Python 3.10+）",
     "ParamSpec 用于装饰器保留函数签名。@functools.wraps 保留元信息但 mypy 不知道。ParamSpec 让 mypy 知道装饰后的函数与原函数签名相同。Concatenate 用于添加参数。ParamSpec 必须配合 Callable[P, R] 使用。"),
    ("Python 中的 enum 的进阶用法：自定义行为",
     "Enum 可添加方法、属性。@unique 防止重复。auto() 自动赋值。可定义 __init__ 添加字段。实例方法/类方法/属性都支持。Enum 继承（不推荐）。@final 装饰器。enum 比 Literal 更强大但更重。"),
    ("Python 的 weakref 模块是什么？什么时候用？",
     "weakref 提供对象的弱引用，不增加引用计数。WeakValueDictionary 值弱引用、WeakKeyDictionary key 弱引用、WeakSet 弱集合。weakref.finalize 对象回收时调用。解决循环引用。缓存场景避免内存泄漏。"),
    ("Python 的 traceback 模块怎么用？",
     "traceback 用于提取、格式化、打印异常堆栈。traceback.print_exc() 打印到 stderr，traceback.format_exc() 返回字符串。logger.exception() 自动记录堆栈。walk_tb() 遍历堆栈。traceback.extract_stack() 获取当前调用栈。"),
    ("Python 中的 zip 函数和 itertools.zip_longest 怎么用？",
     "zip 将多个可迭代对象配对（最短长度），zip_longest 用 fillvalue 填充（最长长度）。zip(*matrix) 矩阵转置。3.10+ 有 strict=True 参数。dict(zip(keys, values)) 创建字典。zip_longest 适合不等长迭代。"),
    ("Python 中元类（Metaclass）如何实现 ORM？",
     "元类可实现 ORM 自动收集字段。Django ORM 和 SQLAlchemy 都用元类实现自动注册。复杂框架用元类，简单项目用 Pydantic/dataclass。生产中 ORM 用成熟库，自造元类 ORM 仅作学习。"),
    ("Python 中的 contextlib 模块怎么用？",
     "contextlib 提供上下文管理器工具。@contextmanager 装饰器把生成器函数转为上下文管理器。ExitStack 动态上下文。suppress 抑制异常。@asynccontextmanager 异步。@redirect_stdout 重定向输出。nullcontext 条件上下文。"),
    ("Python 中的 weakref 和垃圾回收的关系",
     "weakref 不增加引用计数，对象被 GC 时弱引用自动失效。Python GC 用引用计数（主要）和标记-清除（循环引用）。weakref 用于解决循环引用、自动资源清理、缓存。WeakValueDictionary 适合对象缓存。"),
    ("Python 的 functools.reduce 函数怎么用？",
     "reduce 对序列应用二元函数累积计算。reduce(lambda x, y: x + y, [1,2,3]) = 6。可指定 initial 值。可用 operator 模块提高性能。reduce 在 Python 3 仍可用，但很多场景可用 sum、any、all 替代。可读性比 reduce 更高。"),
    ("Python concurrent.futures 的 ThreadPoolExecutor vs ProcessPoolExecutor",
     "ThreadPoolExecutor 用于 I/O 密集（共享内存，受 GIL 限制），ProcessPoolExecutor 用于 CPU 密集（独立进程，无 GIL 限制）。submit 提交任务，as_completed 等待完成，map 批量。线程池简单，进程池启动慢但真正并行。"),
    ("Python 的 dataclasses 如何继承？多态如何实现？",
     "dataclass 支持继承，方法可重写实现多态。多继承需注意字段顺序。@dataclass(frozen=True) 创建不可变子类。抽象基类（ABC）与 dataclass 组合实现接口。Self 类型支持链式调用。"),
    ("Python 的 weakref.WeakValueDictionary 怎么用？",
     "WeakValueDictionary 是值弱引用的字典，对象被 GC 时条目自动删除。用于对象缓存避免内存泄漏。对象需支持 weakref（自定义类默认支持，但 list/dict/int 不支持）。也可使用 finalize 确保资源清理。"),
    ("Python typing 模块：Callable 类型怎么用？",
     "Callable 用于注解可调用对象。Callable[[int], int] 表示接受 int 返回 int。Callable[..., int] 通用（3.10+）。Protocol + Callable 定义回调接口。装饰器工厂用 Callable[P, R] 保留签名。可用于依赖注入、策略模式。"),
    ("Python 的 functools.partial 怎么用？",
     "functools.partial 固定函数部分参数创建新函数。partial(func, exponent=2) 创建 square。partial 保存原函数（func、args、keywords）。partial 比 lambda 更可读。partialmethod 用于方法版。Python 没有内建 curry，但可用 partial 模拟。"),
    ("Python typing 模块：Literal 和 Final 怎么用？",
     "Literal 是字面量类型，限制参数只能是特定值。Literal['GET', 'POST'] 限制为 HTTP 方法。Final 标记不可修改的常量或类。@final 装饰类或方法。LiteralString 防 SQL 注入。Final 用 Final[bool] = True 标记。"),
    ("Python typing 模块：NewType 怎么用？",
     "NewType 创建新类型（运行时仍是基础类型，但 mypy 区分）。UserId = NewType('UserId', int)。强制显式转换 NewType(value)。适合区分语义（ID、邮箱、电话、单位）。Python 3.12+ 推荐用 type 别名。"),
    ("Python 中的 async/await 与 yield from 的关系是什么？",
     "async/await 是 yield from 的语法糖（Python 3.5+）。async def 创建原生协程，@asyncio.coroutine 装饰生成器。await 调用 __await__ 协议。Python 3.4 用 yield from 实现协程，3.5+ 用 async/await 更简洁。"),
    ("Python 中如何处理数据库事务？",
     "数据库事务用 with conn: with cur: 上下文管理器。autocommit 默认 False。异常自动 rollback。SQLAlchemy 2.0 用 session.begin()。asyncpg 用 conn.transaction()。Django 用 @transaction.atomic。savepoint 用于嵌套事务。"),
    ("Python typing 模块：Annotated 怎么用？（Python 3.9+）",
     "Annotated 是类型注解的元数据容器，不影响运行时类型。Annotated[int, Field(gt=0)] 添加验证。FastAPI 用 Annotated 替代 Query/Path/Body 默认值。Pydantic 用 Field 添加验证。文档工具可读取元数据。"),
    ("Python 中的 weakref.finalize 怎么用？",
     "weakref.finalize 在对象被 GC 时自动调用清理函数。weakref.finalize(obj, cleanup_func, *args)。保证调用（即使循环引用）。多个 finalize 按注册顺序调用。可替代 __del__。常用于文件、连接、资源的自动清理。"),
    ("Python typing 模块：TypeGuard 实战（Python 3.10+）",
     "TypeGuard 让 mypy 在调用点收窄类型。def is_str(val: object) -> TypeGuard[str]: return isinstance(val, str)。配合 is_xxx_dict 等自定义验证。Pydantic 提供更完整的验证 + 类型。TypeGuard 轻量，适合简单判断。"),
    ("Python typing 模块的类型系统全景（3.12+ 升级）",
     "3.12+ 新泛型语法：class Stack[T]: pass 代替 Generic[T]。type 别名代替 TypeAlias。Self 类型支持链式调用。@override 装饰器明确重写意图。ParamSpec 简写 **P。TypeVarTuple 简写 *Ts。3.9+ 用内置 list[int] 代替 List[int]。"),
    ("Python typing 模块：常用模式总结与最佳实践",
     "类型注解最佳实践：公共 API 完整注解、装饰器用 ParamSpec、链式 API 用 Self、数据类用 frozen=True、验证用 Pydantic、多态用 Protocol、区分类型用 NewType、常量用 Final、CI 跑 mypy --strict。渐进式添加类型。"),
    ("Python 中 typing 模块：实际项目中的类型注解策略",
     "类型注解策略：按优先级（公共 API → 内部 → 测试）、渐进式迁移、新代码严格旧代码宽松、复杂类型用 TypeGuard + Pydantic、不写类型的场景（动态代码、元编程）、CI 强制检查、文档化类型。"),
    ("Python 中 typing 模块：TypeGuard、Self、TypeVarTuple 综合应用",
     "TypeGuard 自定义类型验证，Self 链式调用，TypeVarTuple 变长泛型。综合应用：Builder 模式用 Self，验证函数用 TypeGuard，泛型装饰器用 TypeVarTuple + ParamSpec。现代 Python 项目充分利用这些特性构建类型安全的 API。"),
    ("Python typing 模块的最佳实践总结（2024）",
     "2024 最佳实践：Python 3.9+ 用内置类型、3.10+ 用 | 联合、3.12+ 用新泛型语法。装饰器必用 ParamSpec + functools.wraps。链式 API 用 Self。Protocol 优于 ABC。TypeGuard 配合 Pydantic。mypy --strict 跑 CI。"),
]

total = len(entries)
print(f"Part 7 entries: {total}")

with open(TARGET, "a", encoding="utf-8") as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"Appended {total} entries")