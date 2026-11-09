"""Exact rendered strings from author gen_gsm.py; no system/format additions."""
from .common import digest

INITIAL = ("Can you solve the following math problem? {} Explain your reasoning. "
           "Your final answer should be a single numerical number, in the form "
           "\\boxed{{answer}}, at the end of your response. ")
REFLECTION = ("Can you double check that your answer is correct. Please reiterate your answer, "
              "with your final answer a single numerical number, in the form \\boxed{{answer}}.")
PEER_PREFIX = "These are the solutions to the problem from other agents: "
PEER = "\n\n One agent solution: ```{}```"
UPDATE = ("\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n "
          "The original math problem is {}. Your final answer should be a single numerical number, "
          "in the form \\boxed{{answer}}, at the end of your response.")


def initial(question):
    return {"role": "user", "content": INITIAL.format(question)}


def update(peers, question):
    if not peers:
        return {"role": "user", "content": REFLECTION}
    return {"role": "user", "content": PEER_PREFIX + "".join(PEER.format(p) for p in peers) + UPDATE.format(question)}


def fingerprint():
    return digest([INITIAL, REFLECTION, PEER_PREFIX, PEER, UPDATE])
