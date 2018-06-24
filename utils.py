import pandas as pd
from textstat.textstat import textstat



def get_smog_score(comment):
    return textstat.text_standard(comment)



def get_average_upvotes(user_id):
    return 0


def create_training_data(comments):
    user_ids = comments["comment_author"].unique()
    user_values = {}
    user_count=0
    for user in user_ids:
        user_count+=1
        print("user: ",user_count,"/",len(user_ids))
        user_comments = comments.loc[comments["comment_author"] == user]
        for index,row in user_comments.iterrows():
            if user in user_values:
                user_value = user_values[user]
                user_value["count"] += 1
                user_value["length_sum"] += len(row["comment_text"])
                user_value["smog_sum"] += get_smog_score(row["comment_text"])
                user_value["upvotes_sum"] += row["TotalVotes"]
                user_values[user] = user_value
            else:
                user_values[user] = {'count': 1, "length_sum": len(row["comment_text"]),
                                          "smog_sum": get_smog_score(row["comment_text"]),
                                          "upvotes_sum": row["TotalVotes"]}

    for user in user_values:
        user_attr=user_values[user]
        user_attr["avg_smog"]=user_attr["smog_sum"]/user_attr["count"]
        user_attr["avg_upvotes"]=user_attr["upvotes_sum"]/user_attr["count"]
        user_attr["avg_length"]=user_attr["length_sum"]/user_attr["count"]
        user_values[user]=user_attr

    return user_values
