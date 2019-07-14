import collections
import random
import copy
import itertools
import pickle
from pathlib import Path
import logging

from anytree import Node
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
from pathlib import Path

logger = logging.getLogger(__file__)



def format_reddit_thing(thing, submission_id):
    if thing["type"] == "submission":
        return "\n".join(
            [thing["title"], thing.get("selftext", "")]
        )
    return thing["body"]


def get_id_for_comments(thing):
    if thing["type"] == "submission":
        return "t3_" + thing["id"]
    else:
        return "t1_" + thing["id"]


def thread2tree(comment_dict, submission):
    """Convert list of reddit comments and a submission to a tree."""
    comment_dict = copy.deepcopy(comment_dict)
    queue = [submission]
    while len(list(itertools.chain(*comment_dict.values()))) > 0:
        for queue_position in range(len(queue) - 1, -1, -1):
            current_id = get_id_for_comments(queue[queue_position])
            found = comment_dict[current_id]
            if len(found):
                break
        next_comment = comment_dict[current_id].pop()
        queue.append(next_comment)


    # convert thread to persona type file. Where persona is submision dict
    submission_id = get_id_for_comments(submission)

    # tree from comments dict
    tree_root = Node(submission_id)
    nodes_by_id = {submission_id: tree_root}
    thing_by_id = {submission_id: submission}
    for thing in queue[1:]:
        parent = nodes_by_id[thing["parent_id"]]
        n = Node(get_id_for_comments(thing), parent=parent)
        nodes_by_id[get_id_for_comments(thing)] = n
        thing_by_id[get_id_for_comments(thing)] = thing
    return nodes_by_id, thing_by_id


def get_dataset(tokenizer, data_path, num_candidates = 3, subreddits=[], max_seq_len=None):
    """
    Load pickle files with reddit comments.

    Structure:
    - valid list
    - dict
        - personality: list[str]
        - utterances: list
            - dict:
                - candidates: list[str]
                - history: list[str]
    
    """
    print('>>get_dataset()')
    data_dir = Path(data_path)
    subreddit_paths = [d for d in data_dir.glob("*/") if d.is_dir()]

    # collect data by subreddit
    splits = dict(train={}, valid={}, test={})
    for subreddit_path in subreddit_paths:
        subreddit_files = sorted(subreddit_path.glob("*.pickle"))
        if len(subreddit_files)>10:
            subreddit = subreddit_path.name
            if (subreddits == []) or (subreddit in subreddits):
                print(f'{len(subreddit_files):10d} threads from /r/{subreddit}')

                # split
                train_files, test_files = train_test_split(
                    subreddit_files, test_size=0.1, random_state=42
                )
                train_files, valid_files = train_test_split(
                    train_files, test_size=0.1, random_state=42
                )

                splits["train"][subreddit] = train_files
                splits["valid"][subreddit] = valid_files
                splits["test"][subreddit] = test_files
            else:
                print(f'subreddit not in filter /r/{subreddit}')
    
    num_train_examples = len(list(itertools.chain(*list(splits["train"].values()))))
    print(f'num_train_examples: {num_train_examples}')
    if len(splits["train"])==0 or num_train_examples< 10:
        raise Exception("not enougth training data found. Check your dataset_path and your --subreddits argument")

    # collect data into the same dict format as hugging face
    dataset2 = collections.defaultdict(list)
    for split, personalities in splits.items():
        total = len(list(itertools.chain(*personalities.values())))
        with tqdm(total=total, desc=f"{split}", unit="file") as prog:
            #key is reddit subthread name, values are pickle files in the folder
            for personality, files in personalities.items():
                utterances = []
                for file in files:
                    #load, eg thread format: {submission: {author: aname, ... score: 42, selftext:
                    # main text body is here}, comment_dict: {[]}}
                    thread = pickle.load(file.open("rb"))
                    prog.update(1)
                    try:
                        nodes_by_id, thing_by_id = thread2tree(
                            thread["comment_dict"], thread["submission"]
                        )
                    except Exception as e:
                        logger.warn("Exception for file '%s', '%s'", file, e)
                        # file.unlink()
                        continue

                    # get utterances
                    # Make a max number of candidates that will fit on your GPU
                    submission_id = get_id_for_comments(thread["submission"])
                    for current_node in nodes_by_id.values():
                        if (
                            current_node.parent
                            and len(current_node.path) > 1
                            and len(current_node.children) >= 1
                        ):
                            history = [
                                format_reddit_thing(thing_by_id[node.name], submission_id)
                                for node in current_node.path
                            ]
                            
                            replies = [
                                thing_by_id[node.name] for node in current_node.children
                            ]

                            # We now want to find distractors. None of these ID's will do
                            correct_ids = [node.name for node in current_node.path] + [submission_id] + [node.name for node in current_node.children]
                            distractor_ids = [k for k, v in nodes_by_id.items() if k not in correct_ids]
                            distractors = [thing_by_id[d_id] for d_id in distractor_ids]

                            # Filter some of the bad data out, yeah it's a hack, but some is very usefull
                            filters = [
                                lambda x: "[deleted]" not in x.get("body", ""),
                                lambda x: "[removed]" not in x.get("body", ""),
                                lambda x: x.get("author", "") != "[removed]",
                                # Filter out the repetative mod and sticky comments
                                lambda x: x.get("author", "") != "AutoModerator",
                                lambda x: not x.get("stickied", False),
                                # Short comments are low information and too easy
                                lambda x: len(x.get("body", "")) > 50,
                            ]
                            # TODO try filtering out replies that overlap too much with history. This avoid repitative qouting and answers
                            for f in filters:
                                replies = filter(f, replies)
                                distractors = filter(f, distractors)

                            # Format "things" from reddit
                            replies = [
                                format_reddit_thing(thing, submission_id)
                                for thing in replies
                            ]
                            distractors = [
                                format_reddit_thing(thing, submission_id)
                                for thing in distractors
                            ]
                            if (len(distractors) >= num_candidates-1):
                                # Distractors at start of candidates, real reply at end
                                for reply in replies:
                                    candidates = random.sample(distractors, num_candidates - 1) + [reply]

                                    utterance = dict(candidates=candidates, history=history)
                                    utterances.append(utterance)

                        else:
                            logger.debug("skipping node with too few paths")
                dataset2[split].append(
                    dict(personality=[personality], utterances=utterances)
                )
                logger.info(f"Utterances for {split} & /r/{personality}: {len(utterances)}")
                if split=='train' and random.random()<0.2:
                    logger.info("Example inputs for %s: %s", personality, utterance)

    logger.info("Tokenize and encode the dataset.")

    max_seq_len = max_seq_len or tokenizer.max_len
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(obj)[:max_seq_len])
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    dataset2 = tokenize(dataset2)
    return dataset2
