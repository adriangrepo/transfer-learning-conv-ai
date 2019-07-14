#!/usr/bin/env python

# coding: utf-8
"""
To get data from pushshift and format it like:

----------

>Could you please share an example how you represented the inputs, including separators?

Sure, here's an example of how I represented the beginning of [this thread in r/math](https://old.reddit.com/r/math/comments/bqjkdb/why_baby_rudin/?sort=top):

    ****S

    Why Baby Rudin?
    Hello all,
    
    I've noticed that Baby Rudin is typically held as the standard for undergraduate real analysis texts. Does anyone know why this is? Is Baby Rudin more rigorous/ comprehensive/ written better than other undergraduate RA texts, or is it just the standard since it's a classic? Just curious.
    ****ES bqjkdb
    
    ****T bqjkdb
    baby rudin is a great analysis textbook for these reasons: 
    
    * it fits a ton of material in a relatively short book
    * proofs are to the point and minimalist, which forces you to do a lot of legwork filling in the details
    * rich and challenging exercises
    
    for reasons why baby rudin is not so loved as an *introductory* analysis textbook, see the above.
    ****ET eo50kel
    
    ****R eo50kel
    I wish I have the money to give you a gold for this amazing comment.
    ****ER eo5jmed
    
    ****R eo5jmed
    save up and spend it on textbooks instead!
    ****ER eo634yh
    
    ****R eo50kel
    See above for reasons.
    See below for proof.
    ****ER eo6toi8
    
    ****T bqjkdb
    Because I experienced the pain, so now it's the next generations turn.
    ****ET eo5mkqq
    
    ****R eo5mkqq
    Exactly, it's a form of ritual hazing for math undergraduates.
    ****ER eo610x7

As you can see, I used the token '****' to represent the beginning/end of comments and submissions. 'S' and 'ES' represent the start and end of submissions, respectively, while 'T' and 'ET' are for top-level comments, and 'R' and 'ER' are for replies (comment-level > 1).

For submissions, the first line is the URL (since this example is a self-post, that line is blank), while the second is the title, and the third is the self-text (if any).

>What hyperparams did you use, especially what context length?

For fine-tuning, I just used the default parameters in the [nshepperd train.py module](https://github.com/nshepperd/gpt-2/blob/finetuning/train.py). 

What exactly are you referring to by "context length"? To generate submissions, I prompt with "****S\n" as the context. For replies, I'd use the entire "ancestry", ie the parent comment, the "grandparent" comment (if applicable), and including the submission info, appended with the correct metadata for the reply.

I'm currently using a temperature of 0.8, and for most of the bots the 'length' parameter is 512 tokens (I use longer lengths for a few of them, like shortscarystories or writingprompts).

>Have you done any cool experiments with these, like making a chat bot, if so, what did you find?

Haven't done a chatbot, but I've been working on a few experiments that are turning out really well so far (IMO). I'm planning on making a post about it this weekend, if I have some free time.


~~ from https://old.reddit.com/r/SubSimulatorGPT2Meta/comments/caelo0/could_you_give_more_details_on_the_input/et8n6xa/?context=10000

Changes:
- [ ] pickle original data so I can reorder on the fly
- [ ] use more info, like Author name? Score
- [ ] write code to concat and split


- psaw rate limite 180/m
- praw rate limite 10 / m but can have multiple active
"""


import argparse
import collections
import copy
import itertools
import logging
import os
import random
import pickle
from pathlib import Path
from psaw import PushshiftAPI
from tqdm import tqdm
import pandas as pd

api = PushshiftAPI()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--out_path",
    type=str,
    default="./data/",
    help="Path or url of the dataset. If empty download from S3.",
)
parser.add_argument(
    "-s",
    "--subreddit",
    type=str,
    action="append",
    default=[
        #"startups",
        "Entrepeneur",
        "tech",
        "TechNewsToday",
        "technews",
        "Cyberpunk",
        "cscareerquestions",
        "technology",
        #"thanksimcured",
        "aspergers",
        #"art",
        "singularity",
        "Showerthoughts",
        "CrazyIdeas",
        "technology",
        "science",
        "askscience",
        "DIY"

    ],
    help="Subreddit names to scrape e.g. ' - s aww - s news '",
)
parser.add_argument(
    "-n",
    "--number_of_threads",
    type=int,
    default=10000,
    help="Number of threads to scrape",
)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


data_dir = Path(args.out_path)


def get_id_for_comments(thing):
    if thing["type"] == "submission":
        return "t3_" + thing["id"]
    else:
        return "t1_" + thing["id"]


def format_comments_dict(comment_dict, submission):
    # Now we want to reconstruct the comment heirachy.
    # 0. Init with the submission in the queue. start with this as target
    # 1. Look at target item in the queue, find it's top rated child comment
    #  1a. If it has one, pop it out, put it at end of queue, go to 1
    #  1b. If it doesn't have comment left, go to previous item in queue
    queue = [submission]
    submission_id = get_id_for_comments(submission)
    while len(list(itertools.chain(*comment_dict.values()))) > 0:
        for queue_position in range(len(queue) - 1, -1, -1):
            current_id = get_id_for_comments(queue[queue_position])
            found = comment_dict[current_id]
            if len(found):
                break
        next_comment = comment_dict[current_id].pop()
        queue.append(next_comment)

    # now format
    text = format_thread(queue, submission_id=submission_id)
    return text


def format_thing(thing, submission_id):
    if thing["type"] == "submission":
        return (
            "****S\n"
            + "\n".join([thing["url"], thing["title"], thing["selftext"]])
            + "\n****ES "
            + thing["id"]
            + "\n"
        )
    elif thing["parent_id"] == submission_id:
        return (
            "****T "
            + thing["parent_id"][3:]
            + "\n"
            + thing["body"]
            + "\n****ET "
            + thing["id"]
            + "\n"
        )
    else:
        return (
            "****R "
            + thing["parent_id"][3:]
            + "\n"
            + thing["body"]
            + "\n****ER "
            + thing["id"]
            + "\n"
        )


def format_thread(queue, submission_id):
    return "\n".join([format_thing(t, submission_id=submission_id) for t in queue])


def psaw_to_dict(thing):
    type_name = type(thing).__name__
    thing = thing.d_
    thing["type"] = type_name
    return thing


def comment_praw2psaw(comment_praw):
    """Convert praw comment to psaw type dict(ish)."""
    cp_dict = copy.deepcopy(comment_praw.__dict__)
    del cp_dict["_reddit"]
    cp_dict["author"] = cp_dict["author"].name
    cp_dict["subreddit"] = cp_dict["subreddit"].name
    cp_dict["parent_id"] = cp_dict["parent_id"][3:]
    return cp_dict


random.shuffle(args.subreddit)
for subreddit in tqdm(args.subreddit, unit="subreddit"):
    try:
        print(subreddit)

        # Since the api often only returns 1000, lets query in monthly intervals
        dates = pd.date_range("2018", "2019", freq="3M")
        date_bins = list(zip(dates[:-1], dates[1:]))
        random.shuffle(date_bins)

        with tqdm(
            desc=subreddit, unit="submission", total=args.number_of_threads
        ) as prog:
            print(subreddit)

            for after, before in date_bins:
                logger.info(
                    "%s",
                    dict(
                        subreddit=subreddit,
                        num_comments=">10",
                        after=after,
                        before=before,
                        sort_type="num_comments",
                    ),
                )

                # Maybe check name is right, and how many threads
                # agg = api.search_submissions(
                #     subreddit=subreddit,
                #     num_comments=">10",
                #     after=after,
                #     before=before,
                #     sort_type="num_comments",
                #     agg="subreddit"
                # )

                submissions = api.search_submissions(
                    subreddit=subreddit,
                    num_comments=">10",
                    after=after,
                    before=before,
                    sort_type="num_comments",
                )
                out_dir = data_dir.joinpath(subreddit)
                os.makedirs(out_dir, exist_ok=True)
                if len(list(out_dir.glob("*.text"))) > args.number_of_threads:
                    print(f"stopping at {args.number_of_threads} threads")
                    break
                for submission in submissions:
                    submission = psaw_to_dict(submission)
                    submission_id = get_id_for_comments(submission)
                    out_file = out_dir.joinpath(submission_id + ".pickle")

                    if not out_file.is_file():
                        # Get comments
                        submission_comment_ids = api._get_submission_comment_ids(
                            submission["id"]
                        )
                        comment_dict = collections.defaultdict(list)

                        # Use psaw
                        try:
                            
                            comments = api.search_comments(ids=submission_comment_ids)
                            # It will just repeat unless we set a limit
                            comments = [
                                next(comments)
                                for _ in tqdm(
                                    range(submission["num_comments"]),
                                    leave=False,
                                    unit="comment",
                                )
                            ]
                            # Or praw... nah slow
                            #             comments = [comment_praw2psaw(reddit.comment(id).refresh()) for id in submission_comment_ids]
                            for comment in comments:
                                comment = psaw_to_dict(comment)
                                comment_dict[comment["parent_id"]].append(comment)

                            # sort by karma, if available
                            for key in comment_dict.keys():
                                comment_dict[key].sort(
                                    key=lambda x: x.get("score", 0), reverse=True
                                )

                            # pickle so we will have original data if wanted, that way we can make changes to input data formatting
                            out_pkl = out_dir.joinpath(submission_id + ".pickle")
                            pickle.dump(
                                dict(submission=submission, comment_dict=comment_dict),
                                out_pkl.open("wb"),
                            )
                            logger.debug("writing pickle %s", out_pkl)

                            # format
                            # text = format_comments_dict(comment_dict, submission)

                            # # write out thread
                            # out_file.write_text(text)
                        except Exception as e:
                            logger.warn(f"Exception {e}, for subreddit={subreddit}, submission_id={submission['id']} submission_comment_ids={submission_comment_ids} after={after} before={before}")
                        prog.update(1)
                    else:
                        logger.debug("skipping existing file %s", out_file)
                        prog.update(1)
    except Exception as e:
        logger.warn(e)
