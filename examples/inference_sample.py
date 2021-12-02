import logging
import re
import time

from plane import CJK

from dbpunctuator.data_process import chinese_split
from dbpunctuator.utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


def chinese_split_2(input):
    """Split Chinese input by:
    - Adding space between every Chinese character. Note: English word will remain as original

    Args:
        input (string): text to apply the regex func to
    """

    regex = re.compile("(?P<%s>%s)" % (CJK.name, CJK.pattern), CJK.flag)
    result = ""
    start = 0
    try:
        for t in regex.finditer(input):
            result += input[start : t.start()]
            result += (
                " "
                + " ".join(
                    [char for char in list(input[t.start() : t.end()]) if char != " "]
                )
                + " "
            )
            start = t.end()
        result += input[start:]
    except TypeError:
        # mal row
        logger.warning(f"error parsing data: {input}")
    return result


if __name__ == "__main__":
    text_1 = "中文测试，english testing url https://sfetsc.com"
    text_2 = "pure english with no Chinese characters"
    text_3 = """
                We shot the scene without a single rehearsal", beatty said.
                As usual the director insisted on a rehearsal, but I convinced him the best opportunity for a realistic battle would be when the two animals first met.
                You simply can't rehearsal a scene like that. Hence the cameramen were ready and the fight was a real one, unfaked...
                And claw to claw, fang to fang battle between natrual enimies of the cat family proved conclusively that the fighting prowess of the lion is superior to that of the tiger according to beatty the tiger lost the battle after a terrific struggle.
                We used a untamed tiger for the battle scene because we figured a good fight was a likely to ensue, the trainer continued.
                That tiger never before been in a cage with a lion.
                Nearly a score of movie stars watched the battle and the majority of them bet on the tiger.
                I had no idea which would win, but I figured sultan had an even chance, though lions are gang fighters and a tiger is supposed to be invinceable in a single-handed battle with almost any animal.
                My reasons for giving the lion an even chance was that I knew that when one takes a hold with his teeth it is in a vital spot, while a tiger sinks his teeth and hangs on whereever he grabs first.
                Thats exactly why tommy lost the fight. While the tiger is simply hanging on to a shoulder, the lion was manuvering into position to get his enemys throat, all the while using his blade-like claws to great advantage, from now on I'll bet on the lion.
                """
    text_4 = """
        2015年12月29号，一南一北两起大案的同日开庭，在年末再次把人们的目光凝聚到司法与公正。在海口，因“事实不清、证据不足”，最高人民法院启动陈满故意杀人、放火案再审，法律将对已经服刑23年的陈满做出公正的判决，人们对于公正的信心来自今天的依法治国；在河北保定，受贿金额超过2亿元的国家能源局煤炭司原副司长魏鹏远出庭受审，依法反腐不留死角，人们对于法可治权的信心同样来自依法治国。
        “让人民群众在每一起司法案件中感受到公平与正义”，这句话，在过去的这一年，前所未有地广为人知、深入人心。
        2015年5月，江苏盐城一对年过七旬的老夫妻因为楼下的污染问题将当地的市场监督管理局告上了法庭。
    """

    start = time.time()
    chinese_split(text_1)
    print(f"text1 split use {(time.time() - start)*1000}")

    start = time.time()
    chinese_split_2(text_1)
    print(f"text1 split2 use {(time.time() - start)*1000}")

    start = time.time()
    chinese_split(text_2)
    print(f"text2 split use {(time.time() - start)*1000}")

    start = time.time()
    chinese_split_2(text_2)
    print(f"text2 split2 use {(time.time() - start)*1000}")

    start = time.time()
    chinese_split(text_3)
    print(f"text3 split use {(time.time() - start)*1000}")

    start = time.time()
    chinese_split_2(text_3)
    print(f"text3 split2 use {(time.time() - start)*1000}")

    start = time.time()
    chinese_split(text_4)
    print(f"text4 split use {(time.time() - start)*1000}")

    start = time.time()
    result = chinese_split_2(text_4)
    print(f"text4 split2 use {(time.time() - start)*1000}")
    print(result)

    from plane import replace

    from dbpunctuator.utils import FULL_URL

    text = "url_test.com"
    start = time.time()
    if re.match(FULL_URL.pattern, text):
        text = "<url>"
    print(f"url replacement use {(time.time() - start) * 1000}")

    start = time.time()
    replace(text, FULL_URL)
    print(f"url replacement use {(time.time() - start) * 1000}")

    text = "url_test"
    start = time.time()
    if re.match(FULL_URL.pattern, text):
        text = "<url>"
    print(f"url replacement use {(time.time() - start)}")
    print(text)

    start = time.time()
    text = replace(text, FULL_URL)
    print(f"url replacement use {(time.time() - start)}")
    print(text)
