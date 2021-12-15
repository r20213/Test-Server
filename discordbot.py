'''
META INFO

Structure for User Storage in Memory:
dict{
    userid: {
        'user': str(message.author.id),
        'prev_context' = [],
        'prev_query' = 'none',
        'count' = 0,
        'prev_doc' = [],
        'prev_links' = [],
        'prev_question':'',
        lang_id:'en'
    }
}

Structure for Support Request History in Memory:
dict{
    userid: {
        'user': str(message.author.id),
        'name': str(bot.fetch_user(message.author.id)),
        'question': [],
        'count': 0
    }
}

Structure for Ratings:
dict{
    userid: {
        'user': str(message.author.id),
        'name': str(bot.fetch_user(message.author.id)),
        'ratings': [int, int, int],
    }
}

Structure for Testing:
dict{
    userid: {
        'user': str(message.author.id),
        'topics': [],
        'overall_grade': [],
        'latest_grade': [],
        'curr_questions': [],
        'curr_gen_answers': [],
        'curr_given_answers': [],
        'current': 0
    }
}

Structure for storing rank related data:
k = 3
dict{
    userid: {
        'user' : str(message.author.id),
        'username' : str,
        'experience' : int, -> Add constant k for every message > 15 chars.
        'interaction' : int, -> Constant k assumed. Once for question, twice for documentation and thrice for videos. TODO Consider time
        'expertise' : int, -> Assume k. Expertise = (Grade/2) + (NumberOfSuccessfulTests)
        'improvement' : int, -> (|Overall - Current|/Overall)*100
        'points_gained' : int, -> (Sum of all)/4
        'level' : int -> Set conditions for Level 1 - 5
    }
}
'''


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


'''
IMPORTS
'''
from PIL import Image, ImageDraw, ImageFont
import io
import aiohttp
from questiongenerator import print_qa
from questiongenerator import QuestionGenerator
import pdfkit
from intents import IntentClassifier
from youtubesearchpython import VideosSearch
from tensorflow.python.keras.models import load_model
import boto3
import spacy
from mrac_qa_v1 import MRAC_QA
from discord.ext import commands
from discord.ext.forms import Form
from discord.ext.commands.converter import IDConverter
from discord import user
import pickle
import re
import os
import discord
from typing import Text
import asyncio
import nest_asyncio
nest_asyncio.apply()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


'''
LOAD UTILITIES, MODELS AND DATA
'''

# AWS Access Keys:
# ID Key - AKIA4XJ3QLA2HISPGYGG
# Secret Key - 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

# Spacy
nlp = spacy.load('en_use_md')

# CDQA Bot
intents = discord.Intents().all()
bot = commands.Bot(command_prefix='!', intents=intents)
mrac = MRAC_QA()
client = boto3.client('translate')

# QG Generator
qg = QuestionGenerator()

# Memory
contextual_memory = {}
request_history = {}
ratings_history = {}
test_history = {}
ranker = {}

# Intents
model = load_model('Intents/models/intents.h5')
with open('Intents/utils/classes.pkl', 'rb') as file:
    classes = pickle.load(file)
with open('Intents/utils/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
with open('Intents/utils/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
nlu = IntentClassifier(classes, model, tokenizer, label_encoder)

print('Setup Complete')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''
HELPER FUNCTION TO PERFORM PERIPHERAL TASKS
'''

# Translation Toolkit


def trans_from(text, lang='en'):
    response = client.translate_text(
        Text=text, SourceLanguageCode='auto', TargetLanguageCode=lang)
    return response['TranslatedText'], response['SourceLanguageCode']


def trans_to(text, source):
    if (source == 'en'):
        return text
    response = client.translate_text(
        Text=text, SourceLanguageCode='auto', TargetLanguageCode=source)
    return response['TranslatedText']


# Helper Function to Calculate Similarity
def check_sim(a1, a2):
    doc_1 = nlp(a1)
    doc_2 = nlp(a2)
    return doc_1.similarity(doc_2)

# Evaluation of Candidate


def evaluate(user_id):

    i = test_history[user_id]['current']
    curr_grade = test_history[user_id]['curr_grade']
    overall_grade = test_history[user_id]['overall_grade']

    print(i)
    print(test_history[user_id]['curr_given_answers'])
    print(test_history[user_id]['curr_gen_answers'])

    while (i != len(test_history[user_id]['curr_given_answers']) and i != len(test_history[user_id]['curr_gen_answers'])):
        a1 = test_history[user_id]['curr_given_answers'][i]
        a2 = test_history[user_id]['curr_gen_answers'][i]

        sim = check_sim(str(a1), str(a2))
        sim = sim*100

        if (curr_grade == -1):
            curr_grade = sim
            if (overall_grade == 0):
                overall_grade = sim
            else:
                overall_grade = (overall_grade + sim)/2
        else:
            curr_grade = (sim + curr_grade)/2
            overall_grade = (overall_grade + sim)/2
        i += 1

    test_history[user_id]['overall_grade'] = overall_grade
    test_history[user_id]['curr_grade'] = curr_grade
    test_history[user_id]['current'] = i

    with open('user_data/test.pickle', 'wb') as handle:
        pickle.dump(test_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return curr_grade, overall_grade

# To get detailed answers (TODO IMPLEMENT)


def get_answers(user_id):
    return

# Rankers


def addxp(user_id, k=3, multiplier=1, identifier='experience'):

    if (str(user_id) not in ranker.keys()):
        ranker[user_id] = {'userid': user_id, 'experience': 0,
                           'improvement': 0, 'expertise': 0, 'points': 0, 'level': 0}

    if (identifier == 'experience'):
        ranker[user_id]['experience'] += k*multiplier
        ranker[user_id]['points'] = round(
            (ranker[user_id]['experience'] + ranker[user_id]['improvement'] + ranker[user_id]['expertise'])/3)

    elif (identifier == 'improvement'):
        try:
            old_grade = test_history[user_id]['overall_grade']
            curr = test_history[user_id]['curr_grade']
            change = (abs(curr - old_grade)/old_grade)*100
            ranker[user_id]['improvement'] += change
        except Exception as e:
            print(e)

    elif (identifier == 'expertise'):
        try:
            nums = len(test_history[user_id]['topic'])
            ranker[user_id]['expertise'] = nums
        except Exception as e:
            print(e)

    with open('user_data/ranker.pickle', 'wb') as handle:
        pickle.dump(ranker, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

# Check Level and update if needed


def checklevel(user_id, author):

    if (user_id not in ranker.keys()):
        print('EXCEPTION - No records exist')
        return

    current_xp = ranker[user_id]['points']
    lvl_start = ranker[user_id]['level']
    lvl_end = int((current_xp ** (1/4)))

    if (lvl_start < lvl_end):
        ranker[user_id]['level'] = lvl_end
        with open('user_data/ranker.pickle', 'wb') as handle:
            pickle.dump(ranker, handle)
        return True, lvl_end
    else:
        return False, lvl_start


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''
Asynchronous Discord Functions
'''


async def add_role(ctx, rolecode):
    memberid = ctx.message.author.id
    print(type(memberid))
    guild = bot.get_guild(915273648166797312)
    role = discord.utils.get(guild.roles, name=rolecode)
    member = await guild.fetch_member(int(memberid))
    await member.add_roles(role)
    return


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


'''
DISCORD APP
'''

# Login as Bot


@bot.event
async def on_ready():
    global contextual_memory
    global ratings_history
    global request_history
    global test_history
    global ranker

    try:
        with open('user_data/ratings.pickle', 'rb') as handle:
            ratings_history = pickle.load(handle)
    except:
        request_history = {}
    try:
        with open('user_data/support.pickle', 'rb') as handle:
            request_history = pickle.load(handle)
        with open('user_data/context.pickle', 'rb') as handle:
            contextual_memory = pickle.load(handle)
            print('SUCCESS')
    except:
        ratings_history = {}
        contextual_memory = {}
        print('FAILURE')
    try:
        with open('user_data/test.pickle', 'rb') as handle:
            test_history = pickle.load(handle)
    except:
        test_history = {}
        print('FAILURE')
    try:
        with open('user_data/ranker.pickle', 'rb') as handle:
            ranker = pickle.load(handle)
    except:
        ranker = {}
        print('FAILURE')
    print('We have logged in as', bot.user.name, bot.user.id)
    guild = bot.get_guild(915273648166797312)
    print(type(guild))
    print(type(await guild.fetch_member(887602345305849867)))
    print(len(guild.members))


# Rank Card
@bot.command()
async def rankcard(ctx, member: discord.Member = None):

    member = member or ctx.author
    user_id = ctx.message.author.id
    guild_id = 915273648166797312
    xp = ranker[str(user_id)]['points']
    level = ranker[str(user_id)]['level']
    int_level = int(level)
    new_level = int_level + 1
    final_xp = new_level**4

    boxes = int((xp/(200*((1/2) * level)))*20)

    tier = 0

    if (level <= 2):
        tier = 3
    elif(level <= 5):
        tier = 2
    else:
        tier = 1

    embed = discord.Embed(title="{}'s current stats.".format(member.name))
    embed.add_field(name=member.name, value=member.mention, inline=True)
    embed.add_field(name="Points", value="{}/{}".format(xp,final_xp), inline=True)
    embed.add_field(name="Level", value=str(int_level), inline=True)
    embed.add_field(name="Tier", value=str(tier), inline=False)
    embed.add_field(name="Progress Bar", value=boxes * ":blue_square:" + (20-boxes) * ":white_large_square:", inline=False)
    embed.set_thumbnail(url=ctx.author.avatar_url)

    await ctx.send(embed=embed)

# Leader Board
@bot.command()
async def leaderboard(ctx, x=10):

    leaderboard = {}
    total = []

    for user in ranker:
        total_amt = ranker[user]['points']
        leaderboard[total_amt] = user
        total.append(total_amt)

    total = sorted(total, reverse=True)

    em = discord.Embed(
        title=f'Top {x} highest leveled members.',
        description='The highest leveled people in this server',
        color=0xa5fb04
    )

    em.set_thumbnail(url="https://media-exp3.licdn.com/dms/image/C560BAQFtJ1qJj7aUdw/company-logo_200_200/0/1623812631226?e=1634169600&v=beta&t=Ep3fAK4IiLtpCqA-hHsrcIBqVnrmpg4rsAlGrjeJmEU")

    index = 1
    for amt in total:
        user = leaderboard[amt]
        lvl = ranker[user]['level']
        curr = await bot.fetch_user(str(user))
        name = curr.name

        em.add_field(name=f'{index}: {name}',
                     value=f'Points - {amt}  |  Level - {lvl}', inline=False)

        if index == x:
            break

        index += 1

    await ctx.send(embed=em)


# Send Questions on a topic
@bot.command()
async def quiz(ctx, topic):

    user_id = str(ctx.message.author.id)
    language = contextual_memory[user_id]['lang_code']

    if (user_id in test_history.keys()):
        if (len(test_history[user_id]['curr_questions']) != 0 and len(test_history[user_id]['curr_given_answers']) != 0 and test_history[user_id]['curr_grade'] == -1):
            await ctx.send(trans_to("You have a previous pending quiz. The scores for those are being calculated and stored. This will be a new test.", language))
            evaluate(user_id)
        if (len(test_history[user_id]['curr_given_answers']) == 0 and len(test_history[user_id]['curr_questions']) != 0):
            await ctx.send(trans_to("You have a previous pending quiz for which you have not attempted any question. That quiz will be deleted. This will be a new test.", language))

        # Store Data
        test_history[user_id]['curr_questions'] = []
        test_history[user_id]['curr_gen_answers'] = []
        test_history[user_id]['curr_given_answers'] = []
        test_history[user_id]['curr_grade'] = -1
        test_history[user_id]['current'] = 0
        if (topic in test_history[user_id]['topic']):
            await ctx.send(trans_to("You have already attempted the test on this topic. Please wait for the intructor to update documents for the same for more questions.", language))
            return
        test_history[user_id]['topic'].append(topic)

        # New Test Generation
        await ctx.send(trans_to('Generating a quiz on the topic - ' + topic + '. This might take a minute. Post the quiz, check your answers. Send you answes in the format - !check "answer1, answer2, answer3"', language))
        answer, context, docname = mrac.discord_query(topic, 3, 1)
        doc = "Data/NLP/" + docname[0]
        with open(doc, 'r') as a:
            article = a.read()
        qa_list = qg.generate(
            article,
            num_questions=5,
            answer_style='all'
        )
        await ctx.send(trans_to('Try and answer these questions:', language))
        qs = ''
        for i in range(len(qa_list)):
            temp = trans_to(qa_list[i]['question'], language) + '\n'
            qs = qs + str(i + 1) + '. ' + temp
            test_history[user_id]['curr_questions'].append(
                qa_list[i]['question'])
            test_history[user_id]['curr_gen_answers'].append(
                qa_list[i]['answer'])
        await ctx.send(qs)
        await ctx.send(trans_to('Use !check to check your answers and grade them. Send you answes in the format - !check "answer1 | answer2 | answer3"', language))

        with open('user_data/test.pickle', 'wb') as handle:
            pickle.dump(test_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        '''
        ADD Expertise
        '''

        addxp(user_id, k=1, multiplier=1, identifier='expertise')
        cond, lvl = checklevel(user_id, ctx.message.author)
        loop = asyncio.get_event_loop()
        if (cond):
            await ctx.author.send('Congratulations! You have levelled up to Level ' + str(lvl) + '. Keep it up!')
            if (lvl <= 2):
                loop.run_until_complete(add_role(ctx, "Tier 3"))
            if (lvl > 2 and lvl <= 5):
                loop.run_until_complete(add_role(ctx, "Tier 2"))
            if (lvl > 5):
                loop.run_until_complete(add_role(ctx, "Tier 1"))

        '''
        End Add Expertise
        '''

    else:
        # First Test
        test_history[user_id] = {'user': str(user_id), 'topic': [], 'curr_questions': [], 'curr_gen_answers': [], 'curr_given_answers': [],
                                 'overall_grade': 0, 'curr_grade': -1, 'current': 0}

        # Store Data
        if (topic in test_history[user_id]['topic']):
            await ctx.send(trans_to("You have already attempted the test on this topic. Please wait for the intructor to update documents for the same for more questions.", language))
            return
        test_history[user_id]['topic'].append(topic)

        # New Test Generation
        await ctx.send(trans_to('Generating a quiz on the topic - ' + topic + '. This might take a minute. Post the quiz, check your answers. Send you answes in the format - !check "answer1 | answer2 | answer3"', language))
        answer, context, docname = mrac.discord_query(topic, 3, 1)
        doc = "Data/NLP/" + docname[0]
        with open(doc, 'r') as a:
            article = a.read()
        qa_list = qg.generate(
            article,
            num_questions=5,
            answer_style='all'
        )
        await ctx.send(trans_to('Try and answer these questions:', language))
        qs = ''
        for i in range(len(qa_list)):
            temp = trans_to(qa_list[i]['question'], language) + '\n'
            qs = qs + str(i + 1) + '. ' + temp
            test_history[user_id]['curr_questions'].append(
                qa_list[i]['question'])
            test_history[user_id]['curr_gen_answers'].append(
                qa_list[i]['answer'])
        await ctx.send(qs)
        await ctx.send(trans_to('Use !check to check your answers and grade them. Send you answes in the format - !check "answer1 | answer2 | answer3"', language))

        with open('user_data/test.pickle', 'wb') as handle:
            pickle.dump(test_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        '''
        ADD Expertise
        '''

        addxp(user_id, k=1, multiplier=1, identifier='expertise')
        cond, lvl = checklevel(user_id, ctx.message.author)
        loop = asyncio.get_event_loop()
        if (cond):
            await ctx.author.send('Congratulations! You have levelled up to Level ' + str(lvl) + '. Keep it up!')
            if (lvl <= 2):
                loop.run_until_complete(add_role(ctx, "Tier 3"))
            if (lvl > 2 and lvl <= 5):
                loop.run_until_complete(add_role(ctx, "Tier 2"))
            if (lvl > 5):
                loop.run_until_complete(add_role(ctx, "Tier 1"))

        '''
        End Add Expertise
        '''

    return


# Send Answers on a topic
@bot.command()
async def answer(ctx):

    user_id = str(ctx.message.author.id)
    language = contextual_memory[user_id]['lang_code']

    if (len(test_history[user_id]['curr_questions']) == 0):
        await ctx.send(trans_to('Take a quiz first!', language))
        return

    # Pending - Code to add to memory
    ans = ''
    for i in range(len(test_history[user_id]['curr_gen_answers'])):
        temp = trans_to(test_history[user_id]
                        ['curr_gen_answers'][i], language) + '\n'
        ans = ans + str(i + 1) + '. ' + temp
    await ctx.send(ans)
    await ctx.send(trans_to('Use !check to check your answers and grade them. Send you answes in the format - !check "answer1, answer2, answer3"', language))

    return


# Check Answers on a topic
@bot.command()
async def check(ctx, list_of_answers):

    user_id = str(ctx.message.author.id)
    language = contextual_memory[user_id]['lang_code']

    if (len(test_history[user_id]['curr_questions']) == 0):
        await ctx.send(trans_to('Take a quiz first!', language))
        return

    if (test_history[user_id]['current'] == len(test_history[user_id]['curr_gen_answers'])):
        await ctx.send(trans_to('All questions have been evaluated. You current grade is ' + str(round(test_history[user_id]['curr_grade'], 2)) + '%.', language))
        test_history[user_id]['curr_grade'] = -1
        return

    lis = list_of_answers.split(',')

    for i in lis:
        test_history[user_id]['curr_given_answers'].append(trans_from(i)[0])

    curr, overall = evaluate(user_id)

    curr = round(curr, 2)
    overall = round(overall, 2)

    await ctx.send(trans_to('Your current grade for this test is ' + str(curr) + '%. Your cumulative grade is ' + str(overall) + '%.', language))

    '''
    ADD Improvement
    '''

    addxp(user_id, k=1, multiplier=1, identifier='improvement')
    cond, lvl = checklevel(user_id, ctx.message.author)
    loop = asyncio.get_event_loop()
    if (cond):
        await ctx.author.send('Congratulations! You have levelled up to Level ' + str(lvl) + '. Keep it up!')
        if (lvl <= 2):
            loop.run_until_complete(add_role(ctx, "Tier 3"))
        if (lvl > 2 and lvl <= 5):
            loop.run_until_complete(add_role(ctx, "Tier 2"))
        if (lvl > 5):
            loop.run_until_complete(add_role(ctx, "Tier 1"))

    '''
    End Add Improvement
    '''

    return


# Main
@bot.event
async def on_message(message):

    global contextual_memory
    global request_history
    global ratings_history

    # Check if bot is not the author.
    # Also check if command.
    if message.author == bot.user:
        return
    if (message.content[:1] == '!'):
        await bot.process_commands(message)
        return

    # Details of Message
    user_id = str(message.author.id)
    print(user_id, message.content)
    q = message.content

    # If User in memory - Use Contextual Memory History
    if (user_id in contextual_memory.keys()):
        ctx = await bot.get_context(message)
        print('Found Memory')

        # Get language and remember it
        lang_corrector = False
        if (q[-1:] == '?'):
            lang_corrector = True
        q, lang = trans_from(q)
        if (lang_corrector and q[-1:] != '?'):
            q = q + '?'

        # If Rating is accepted
        if (q == "rate"):
            userid = str(message.author.id)

            form = Form(ctx, trans_to('Survey', lang))
            form.add_question(
                trans_to('Rate my accuracy from 1-5', lang), 'first')
            form.add_question(
                trans_to('Rate my speed from 1-5', lang), 'second')
            form.add_question(
                trans_to('Rate the documents/videos retrieved from 1-5', lang), 'third')
            result = await form.start()

            user = await bot.fetch_user(userid)
            name = user.name

            if (userid in ratings_history.keys()):
                ratings_history[userid]['ratings'] = [
                    int(result.first), int(result.second), int(result.third)]
            else:
                ratings_history[userid] = {'user': userid, 'name': name, 'ratings': [
                    int(result.first), int(result.second), int(result.third)]}

            print(result)
            await user.send(trans_to('Your ratings are registered. We will get back to you soon!', lang))

            return

        # Callback for Rating
        if (q.isdigit()):
            return

        # If asking for support
        if (q == "support"):
            question_asked = contextual_memory[user_id]['prev_query']
            user = await bot.fetch_user(user_id)
            lang = contextual_memory[user_id]['lang_code']
            boss = await bot.fetch_user("887602345305849867")
            name = user.name

            mail = "Greeting, \nUser: " + "<@!" + user_id + ">" + " is requesting help with a question I could not answer. Question is shown below: \n**" + \
                question_asked + "**\nYou are requested to reach out to them at the earliest. Please use translate to convert to language code - " + lang + "."

            await boss.send(mail)
            await message.channel.send('Your request is sent!')

            # If previous requests
            if (user_id in request_history.keys()):
                request_history[user_id]['count'] += 1
                request_history[user_id]['question'].append(question_asked)
            # Else
            else:
                request_history[user_id] = {
                    'userid': user_id, 'name': name, 'question': [], 'count': 0}
                request_history[user_id]['count'] += 1
                request_history[user_id]['question'].append(question_asked)

            with open('user_data/support.pickle', 'wb') as handle:
                pickle.dump(request_history, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

            return

        # If no more help required. CTA Form.
        if (q == "no" or q == "No"):
            await message.channel.send('Glad you are satisfied! To make me better, you can choose to rate my capabilites by typing **rate**')
            return

        # Embed Videos
        if (q == "videos" or q == "video" or q == "Video"):

            '''
            Add XP - Is a video
            '''
            addxp(user_id, 3, 3, identifier='experience')
            cond, lvl = checklevel(user_id, message.author)
            mention = message.author.mention
            loop = asyncio.get_event_loop()
            if (cond):
                await ctx.author.send('Congratulations! You have levelled up to Level ' + str(lvl) + '. Keep it up!')
                if (lvl <= 2):
                    loop.run_until_complete(add_role(ctx, "Tier 3"))
                if (lvl > 2 and lvl <= 5):
                    loop.run_until_complete(add_role(ctx, "Tier 2"))
                if (lvl > 5):
                    loop.run_until_complete(add_role(ctx, "Tier 1"))
            '''
            End Add XP
            '''

            if (contextual_memory[user_id]['prev_query'] == 'none'):
                await message.channel.send(trans_to("Ask a question first", lang))
                return
            reply = "**Here are some helpful videos:** \n" + '1. ' + \
                contextual_memory[user_id]['prev_links'][0] + ' \n' + '2. ' + \
                    contextual_memory[user_id]['prev_links'][1] + ' \n' + \
                '3. ' + contextual_memory[user_id]['prev_links'][2]
            await message.channel.send(reply)
            return

        # Embed Context
        if (q == 'Yes' or q == "yes" or q == "More" or q == "more"):

            '''
            Add XP - Is context
            '''
            addxp(user_id, 3, 1, identifier='experience')
            cond, lvl = checklevel(user_id, message.author)
            mention = message.author.mention
            loop = asyncio.get_event_loop()
            if (cond):
                await ctx.author.send('Congratulations! You have levelled up to Level ' + str(lvl) + '. Keep it up!')
                if (lvl <= 2):
                    loop.run_until_complete(add_role(ctx, "Tier 3"))
                if (lvl > 2 and lvl <= 5):
                    loop.run_until_complete(add_role(ctx, "Tier 2"))
                if (lvl > 5):
                    loop.run_until_complete(add_role(ctx, "Tier 1"))
            '''
            End Add XP
            '''

            if (contextual_memory[user_id]['prev_query'] == 'none'):
                if (lang == 'en'):
                    await message.channel.send("Ask a question first.")
                    return
                else:
                    await message.channel.send(trans_to("Ask a question first.", lang))
                    return
            if (contextual_memory[user_id]['count'] > 2):
                if (lang == 'en'):
                    await message.channel.send("That is all the information I have.")
                    return
                else:
                    await message.channel.send(trans_to("That is all the information I have.", lang))
                    return

            reply = "**Here is some context:** \n\n" + \
                trans_to(contextual_memory[user_id]['prev_context']
                         [contextual_memory[user_id]['count']], lang)
            contextual_memory[user_id]['count'] += 1
            await message.channel.send(reply)
            await message.channel.send(trans_to('Some more information? Just ask me for more or you could also ask for the complete document by asking me for docs.', lang))
            return

        # Downloadable Content
        if (q == 'documents' or q == 'docs' or q == 'Documents' or q == 'Docs'):

            '''
            Add XP - Is a document
            '''
            addxp(user_id, 3, 2, identifier='experience')
            cond, lvl = checklevel(user_id, message.author)
            mention = message.author.mention
            loop = asyncio.get_event_loop()
            if (cond):
                await ctx.author.send('Congratulations! You have levelled up to Level ' + str(lvl) + '. Keep it up!')
                if (lvl <= 2):
                    loop.run_until_complete(add_role(ctx, "Tier 3"))
                if (lvl > 2 and lvl <= 5):
                    loop.run_until_complete(add_role(ctx, "Tier 2"))
                if (lvl > 5):
                    loop.run_until_complete(add_role(ctx, "Tier 1"))
            '''
            End Add XP
            '''

            if (contextual_memory[user_id]['prev_query'] == 'none'):
                await message.channel.send(trans_to("Ask a question first", lang))
                return
            reply = "**Document Related to Question** \n\n"
            await message.channel.send(reply)
            documents = contextual_memory[user_id]['prev_doc']

            for i in range(len(documents)):

                path = documents[i]
                print(path)
                output = "documents/" + \
                    str(i+1) + ".txt"

                with open(path) as f, open(output, 'w') as f2:
                    for x in f:
                        if (not x.strip().replace("\n", "")):
                            f2.write(x)
                            continue
                        newline = x.strip()+'\n'
                        if len(newline.rstrip()) >= 3:
                            f2.write(newline)

                with open(output, 'r+') as f:
                    content = f.read()
                    content = re.sub(r'\n\s*\n', '\n\n', content)
                    f.truncate(0)
                    f.seek(0)
                    f.write(content)

                with open(output) as file:
                    with open("documents/" + str(i+1) + ".html", "w") as output:
                        file = file.read()
                        file = file.replace("\n", "<br>")
                        output.write(file)

                to_write = "documents/FILE" + \
                    str(i + 1) + ".pdf"
                pdfkit.from_file(
                    "documents/" + str(i+1) + ".html", to_write)

                await message.channel.send(file=discord.File(to_write))

            await message.channel.send(trans_to('Too much to read? Ask me for videos!', lang))
            return

        # In Case of Help - List Commands
        if (q == 'help'):
            helper = 'For more information regarding a question type - **more** \nFor a explanatory document - **docs** \nFor videos - **videos**.'
            await message.channel.send(trans_to(helper, lang))
            return

        # Intents for Greet, Thanks and Bye. CTA to support and survey.
        if (q[-1:] != '?'):
            text, z = nlu.get_intent(q)
            if (text == 'greeting'):
                reply_to_new_user = 'Hi! Great to see you again!'
                if (lang == 'en'):
                    await message.channel.send(reply_to_new_user)
                else:
                    await message.channel.send(trans_to(reply_to_new_user, lang))
            elif (text == 'thank_you'):
                reply_to_new_user = 'Glad you are satisfied! To make me better, you can choose to rate my capabilites by typing **rate**'
                if (lang == 'en'):
                    await message.channel.send(reply_to_new_user)
                else:
                    await message.channel.send(trans_to(reply_to_new_user, lang))
            else:
                reply_to_new_user = 'See you soon! To make me better, you can choose to rate my capabilites by typing **rate**'
                if (lang == 'en'):
                    await message.channel.send(reply_to_new_user)
                else:
                    await message.channel.send(trans_to(reply_to_new_user, lang))
            return

        '''
        Add XP - Is a question
        '''
        addxp(user_id, 3, 1, identifier='experience')
        checklevel(user_id, message.author)
        cond, lvl = checklevel(user_id, message.author)
        loop = asyncio.get_event_loop()
        if (cond):
            await ctx.author.send('Congratulations! You have levelled up to Level ' + str(lvl) + '. Keep it up!')
            if (lvl <= 2):
                loop.run_until_complete(add_role(ctx, "Tier 3"))
            if (lvl > 2 and lvl <= 5):
                loop.run_until_complete(add_role(ctx, "Tier 2"))
            if (lvl > 5):
                loop.run_until_complete(add_role(ctx, "Tier 1"))
        '''
        End Add XP
        '''

        # In case reached here, use MRAC to answer question.
        contextual_memory[user_id]['all_questions'].append(q)

        if (lang == 'en'):
            await message.channel.send("I am thinking of an answer.")
        else:
            await message.channel.send(trans_to("I am thinking of an answer.", lang))

        answer, context, docname = mrac.discord_query(q, 3, 1)
        docs_path = []
        for i in docname:
            docs_path.append("saved_docs/" + i)
            print(i)

        contextual_memory[user_id]['prev_query'] = q
        contextual_memory[user_id]['prev_context'] = context
        contextual_memory[user_id]['count'] = 0
        contextual_memory[user_id]['prev_doc'] = docs_path

        if (lang == 'en'):
            reply = "**Here are some possible answers:** \n" + '1. ' + \
                answer[0] + ' \n' + '2. ' + \
                    answer[1] + ' \n' + '3. ' + answer[2]
            await message.channel.send(reply)
        else:
            helper = trans_to("Here are some possible answers:", lang)
            answer_0 = trans_to(answer[0], lang)
            answer_1 = trans_to(answer[1], lang)
            answer_2 = trans_to(answer[2], lang)
            reply = "**" + helper + "** \n" + '1. ' + answer_0 + \
                ' \n' + '2. ' + answer_1 + ' \n' + '3. ' + answer_2
            await message.channel.send(reply)

        videosSearch = VideosSearch(q, limit=3)
        d = videosSearch.result()
        links = []
        for i in d['result']:
            links.append(i['link'])

        contextual_memory[user_id]['prev_links'] = links

        if (lang != contextual_memory[user_id]['lang_code']):
            contextual_memory[user_id]['lang_code'] = lang

        with open('user_data/context.pickle', 'wb') as handle:
            pickle.dump(contextual_memory, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        return

    # If User is new. Create Memory Block and go from there.
    else:

        q, lang = trans_from(q)
        ctx = await bot.get_context(message)

        # Create User in memory
        contextual_memory[user_id] = {'user': str(message.author.id), 'prev_context': [
        ], 'prev_query': 'none', 'count': 0, 'prev_doc': [], 'prev_links': [], 'all_questions': [], 'lang_code': lang}

        if (q == "no" or q == "No" or q == 'Yes' or q == "yes" or q == "More" or q == "more" or q == "videos" or q == "video" or q == "Video" or q == 'documents' or q == 'docs' or q == 'Documents' or q == 'Docs' or q == 'support' or q == 'rate'):
            nomemory_base = "I can't seem to find you in my memory and so cannot help you with this. Hope you don't mind asking the question again."
            if (lang == 'en'):
                await message.channel.send(nomemory_base)
            else:
                nomemory_base_trans = trans_to(nomemory_base, lang)
                await message.channel.send(nomemory_base_trans)
            return

        if (q == 'help'):
            helper = 'For more information regarding a question type - **more** \nFor a explanatory document - **docs** \nFor videos - **videos**.'
            if (lang == 'en'):
                await message.channel.send(helper)
            else:
                helper_trans = trans_to(helper, lang)
                await message.channel.send(helper_trans)
            return

        if (q[-1:] != '?'):
            text, z = nlu.get_intent(q)
            if (text == 'greeting'):
                reply_to_new_user = 'Hi! I can answer your questions on NLP. I see you are new here! To ask me a question, just type it out and add a question mark. If you need help with my working just typr **help**. `In case you are not satisfied by my answers, feel free to ask for live support by typing **support**`.'
                if (lang == 'en'):
                    await message.channel.send(reply_to_new_user)
                else:
                    await message.channel.send(trans_to(reply_to_new_user, lang))
            elif (text == 'thank_you'):
                reply_to_new_user = 'Glad you are satisfied! To make me better, you can choose to rate my capabilites by typing **rate**'
                if (lang == 'en'):
                    await message.channel.send(reply_to_new_user)
                else:
                    await message.channel.send(trans_to(reply_to_new_user, lang))
            else:
                reply_to_new_user = 'See you soon! To make me better, you can choose to rate my capabilites by typing **rate**'
                if (lang == 'en'):
                    await message.channel.send(reply_to_new_user)
                else:
                    await message.channel.send(trans_to(reply_to_new_user, lang))
            return

        '''
        Add XP - Is a question
        '''
        addxp(user_id, 3, 1, identifier='experience')
        cond, lvl = checklevel(user_id, message.author)
        mention = message.author.mention
        loop = asyncio.get_event_loop()
        if (cond):
            await ctx.author.send('Congratulations! You have levelled up to Level ' + str(lvl) + '. Keep it up!')
            if (lvl <= 2):
                loop.run_until_complete(add_role(ctx, "Tier 3"))
            if (lvl > 2 and lvl <= 5):
                loop.run_until_complete(add_role(ctx, "Tier 2"))
            if (lvl > 5):
                loop.run_until_complete(add_role(ctx, "Tier 1"))
        '''
        End Add XP
        '''

        contextual_memory[user_id]['all_questions'].append(q)

        if (lang == 'en'):
            await message.channel.send("I am thinking of an answer.")
        else:
            await message.channel.send(trans_to("I am thinking of an answer.", lang))

        answer, context, docname = mrac.discord_query(q, 3, 1)
        docs_path = []
        for i in docname:
            docs_path.append("saved_docs/" + i)
            print(i)

        contextual_memory[user_id]['prev_query'] = q
        contextual_memory[user_id]['prev_context'] = context
        contextual_memory[user_id]['count'] = 0
        contextual_memory[user_id]['prev_doc'] = docs_path

        if (lang == 'en'):
            reply = "**Here are some possible answers:** \n" + '1. ' + \
                answer[0] + ' \n' + '2. ' + \
                    answer[1] + ' \n' + '3. ' + answer[2]
            await message.channel.send(reply)
        else:
            helper = trans_to("Here are some possible answers:", lang)
            answer_0 = trans_to(answer[0], lang)
            answer_1 = trans_to(answer[1], lang)
            answer_2 = trans_to(answer[2], lang)
            reply = "**" + helper + "** \n" + '1. ' + answer_0 + \
                ' \n' + '2. ' + answer_1 + ' \n' + '3. ' + answer_2
            await message.channel.send(reply)

        videosSearch = VideosSearch(q, limit=3)
        d = videosSearch.result()
        links = []
        for i in d['result']:
            links.append(i['link'])

        contextual_memory[user_id]['prev_links'] = links

        if (lang != contextual_memory[user_id]['lang_code']):
            contextual_memory[user_id]['lang_code'] = lang

        with open('user_data/context.pickle', 'wb') as handle:
            pickle.dump(contextual_memory, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        await bot.process_commands(message)

        return


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Essential One time code - DON'T CHANGE - Restricted
bot.run('OTE1MjcyNDc0NzE3OTk1MDI5.YaZL6g.GigaS7cyLzh-fyL35WEG7ksCBEU')
