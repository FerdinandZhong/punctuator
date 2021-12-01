import logging

from dbpunctuator.inference import Inference, InferenceArguments
from dbpunctuator.utils import ALL_PUNCS, DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP
from dbpunctuator.utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


def produce_sample_text(text, repl=None):
    puncs = dict(zip(ALL_PUNCS, [repl] * len(ALL_PUNCS)))
    return text.lower().translate(puncs)


news_text = [
    """该报道指日本政府最快将在周一下午做出宣布。

    日本首相岸田文雄同日早前告诉媒体，面对奥密克戎新变种毒株，日本政府“有强烈的危机感”，并表示考虑采取进一步的措施以加强边境管制。

    日本政府刚在星期天宣布，禁止来自非洲南部九个国家包括南非的外国人入境。
    """,  # noqa: E501
    """
    台军方的统计称，28日解放军有空警—500机2架次、运—9机1架次、轰—6战机5架次、运油—20一架次、歼—10战机6架次、歼—11战机4架次、歼—16战机8架次进入西南防空识别区，
    其中台空军首度侦测掌握运油—20绕台。
    中时电子报称，运油—20机型于2019年起试飞，并开始密集测试，对象包含歼—20，截至今年11月至少服役5架，是目前大陆最大型的军用飞机。
    """,  # noqa: E501
]

long_text = [
    """
    作为引领华为在“无人区”探索的年轻队伍，天才少年在内部被视为攻克世界难题的冲锋队。

    按照华为此前公开的邮件，天才少年的工资按年度工资制度发放，共有三档，分别为89.6万－100.8万元、140.5万－156.5万元、182万－201万元。

    虽然提出了具有吸引力的薪资，但华为“天才少年”的招聘标准非常严格，一般需要经历7轮左右流程：简历筛选、笔试、初面、主管面试、若干部长面试、总裁面试、HR面试。任何一个环节出现问题或表现不佳都有可能失去进入华为的机会。

    钟钊则是在“天才少年”计划中拿到最高档位的员工。

    公开资料显示，钟钊出生于1991年，本科就读于华中科技大学的软件工程专业。刚上大三的他，就在2012全国大学生数模竞赛中获得了湖北一等奖。而后前往中国科学院大学自动化研究所攻读硕士、博士，硕博阶段攻读专业都是“模式识别与智能系统”。

    在自述中，钟钊提到受父亲影响，在幼儿园和小学阶段就已接触基础编程。

    “我父亲是很早一批的北大学生，之前在中科院的高能物理所（原子能研究所）做研究，是钱三强何泽慧夫妇的学生，他最早是搞核物理研究，就是做两弹一星，主要是氢弹的研究。他们需要用计算机来进行核物理相关的计算，后来又转到计算机领域的研究了。父亲平时工作中有些东西是我可以接触到的，我还记得去机房和办公室找他，得穿鞋套来避免静电。等我父亲下班的过程我拿空闲的电脑玩小乌龟画画（LOGO语言），在DOS系统里寻找大学生上课时偷偷安装的新游戏。所以我的兴趣就是在父亲的潜移默化中培养起来了。”

    2018年，钟钊去美国参加CVPR（IEEE国际计算机视觉与模式识别会议），偶然得知华为在布局AutoML（Automated Machine Learning）技术。强有力的算力和平台支持以及真实的业务场景吸引了钟钊加入华为。
    """  # noqa: E501
]

if __name__ == "__main__":
    args = InferenceArguments(
        model_name_or_path="models/chinese_punctuator",
        tokenizer_name="bert-base-chinese",
        tag2punctuator=DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP,
    )

    inference = Inference(inference_args=args, verbose=False)

    logger.info(
        f"testing result {inference.punctuation([produce_sample_text(text) for text in news_text])}"
    )

    logger.info(
        f"testing result {inference.punctuation([produce_sample_text(text) for text in long_text])}"
    )
