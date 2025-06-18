answer_prefix_extraction_prompt = """\
You task is to extract a prefix from an article that satisfies some criterion which we will describe later. Specifically, you'll be given a wikipedia ARTICLE enclosed in triple backticks ```...```. This article is a reference for a factoid question that asks about the identity of an OBJECT with RELATION to SUBJECT. You will be provided the SUBJECT and any possible aliases, as well as the OBJECT and any possible aliases. You will also be provided the RELATION. You will be provided the QUESTION. And possible ANSWERS (which are essentially aliases to OBJECT).

Start by expanding the entity or concepts in ANSWERS to include additionally those concepts or entities that 1) appear in the ARTICLE 2) is equivalent to ANSWER or is a more specific version of ANSWER. These should be really short phrases. If there is a longer phrase, you should break it up into component entities or concepts. For example, "United States" is more specific than "North America", "Shanghai" is more specific than "China", and "Writes Music" can be equivalent or more specific to "Composer", depending on the context.

Next, extract the longest prefix string of the article (it can stop in the middle of a sentence, and often should, in order to be as long as possible) right before the first expanded ANSWER phrase. Explicitly explain how you checked that 1) doesn't include any ANSWER or expanded ANSWER and 2) it is the longest possible string prefix satisfying 1. If you realize you made a mistake, add a revised prefix at the end, also enclosed in <prefix></prefix>.

Finally, it is very important that you extract an EXACT prefix of the ARTICLE (meaning even if there's a typo, you preserve it exactly, and preserve any whitespace, if there are multiple newlines, keep it, don't combine it into one). Also, make sure it is actually a PREFIX - that is, it starts from the title line.
"""

subject_prefix_extraction_prompt = """\
You task is to extract a prefix from an article that satisfies some criterion which we will describe later. Specifically, you'll be given a wikipedia ARTICLE enclosed in triple backticks ```...```. This article is a reference for a factoid question that asks about the identity of an OBJECT with RELATION to SUBJECT. You will be provided the SUBJECT and any possible aliases, as well as the OBJECT and any possible aliases. You will also be provided the RELATION. You will be provided the QUESTION. And possible ANSWERS (which are essentially aliases to OBJECT).

Extract the longest prefix string of the article (it can stop in the middle of a sentence, and often should, in order to be as long as possible) right before the first content word that discusses the SUBJECT. Explicitly explain how you checked that 1) it doesn't include any content discussion of the SUBJECT 2) it is the longest possible string prefix satisfying 1. Make sure it's actually as long as possible. It is often possible to include the title and a few words of the first sentence. If you realize you made a mistake, add a revised prefix at the end, also enclosed in <prefix></prefix>.

It is important that you extract an exact prefix of the ARTICLE (meaning even if there's a typo, you copy it exactly, and copy any whitespace).
"""
