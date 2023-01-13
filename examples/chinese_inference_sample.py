import logging

from punctuator.inference import Inference, InferenceArguments
from punctuator.utils import ALL_PUNCS, DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP
from punctuator.utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


def produce_sample_text(text, repl=None):
    puncs = dict(zip(ALL_PUNCS, [repl] * len(ALL_PUNCS)))
    return text.lower().translate(puncs)


news_text = [
    """
    近日，香港大学工程学院机械工程系黄明欣教授领导的团队，与港大李嘉诚医学院免疫与感染研究中心潘烈文教授的研究团队合作，发明了一种高铜含量的不锈钢，能成功杀灭存留于其表面包括新冠病毒在内的抗病原体。
    """,  # noqa: E501
]

long_text = [
    """
    首节比赛，詹姆斯突破首开纪录，随后塔克连得5分，湖人7-0开局。瓦格纳三分破冰，詹姆斯转换打停魔术。随后科尔-安东尼送上三分，瓦格纳勾手打成，哈里斯三分再中，小洛二次进攻打成，魔术轰出一波10-0，湖人9-13落后果断叫停。霍华德二次进攻止血，罗斯和科尔-安东尼连取4分。安东尼跳投回敬，奥克克上篮打成，罗斯和安东尼对飚三分，汉普顿再中三分，魔术25-18领先进入次节。


    次节伊始，小卡特跳投得手，威少突破回敬，安东尼两罚全中又送上三分，班巴跳投打成，魔术31-25领先。詹姆斯送钉板大帽又打成2+1，里夫斯两罚全中，小塔克送暴扣，湖人一波7-0打停魔术。科尔-安东尼和艾灵顿对飚三分，卡特跳投打成，魔术36-35领先。随后卡特命中三分，威少连得4分，洛佩兹扣篮打进，罗斯命中三分，布拉德利回敬三分球，科尔-安东尼三分也中，塔克上篮得分，罗斯三罚全中，詹姆斯连得4分，半场结束魔术52-49领先。

    易边再战场上风云突变，詹姆斯三分追平比分，小瓦格纳突破抛投止血，詹姆斯三分再中，卡特两罚全中，魔术56-55领先。湖人突然发力，詹姆斯跳投得手又送上暴扣，塔克连中三分，霍华德和小乔丹连送暴扣，湖人轰出一波23-0一举建立22分优势。罗斯两罚终于破荒，随后湖人再度连取4分，小洛抛投打成，安东尼三分命中，小瓦格纳两罚全中，湖人85-62领先23分进入末节。


    末节决战，罗斯和威少互送突破，随后小乔丹起飞打成，卡特回敬补篮，小瓦格纳送上三分，汉普顿暴扣得手，魔术追至71-89。塔克放篮打成，魔术连续取分轰出一波9-3将分差迫近到14分，湖人赶紧叫停。随后卡特扣篮得分，詹姆斯命中三分，加里-哈里斯命中三分，魔术将分差缩小至10分。关键时刻詹姆斯助攻里夫斯命中三分，詹姆斯上篮打进，比赛失去悬念，湖人锁定胜利。
    """  # noqa: E501
]

if __name__ == "__main__":
    args = InferenceArguments(
        model_weight_name="Qishuai/distilbert_punctuator_zh",
        tokenizer_name="Qishuai/distilbert_punctuator_zh",
        tag2punctuator=DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP,
    )

    inference = Inference(inference_args=args, verbose=False)

    logger.info(
        f"testing result {inference.punctuation([produce_sample_text(text) for text in news_text])}"
    )

    logger.info(
        f"testing result {inference.punctuation([produce_sample_text(text) for text in long_text])}"
    )
