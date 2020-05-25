import re
import config
import logging
import string
import os
import requests
import json
from telegram.ext import *
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove

HOST = "covid-fake-news-backend.herokuapp.com"
# HOST = "0.0.0.0:8080"
TOKEN = config.TOKEN
#PORT = int(os.environ.get('PORT', '8443'))
bot = Bot(token=TOKEN)
loaded_json = {}

# Set up basic logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# HELPER FUNCTIONS


def getPercentage(number, total):
    return number/total * 100.0

# API FUNCTIONS


def getCheckFakeNewsResults(host, update):
    print(update)
    print(type(update))
    headers = {'Content-type': 'application/json', 'Accept': '*/*'}
    response = requests.post('http://' + host + '/checkFakeNews',
                             json={"boomerText": update.message.text}, headers=headers)
    print()
    return response.json()


def updateVote(host, isTrueVote):
    headers = {'Content-type': 'application/json', 'Accept': '*/*'}
    response = requests.post('http://' + host + '/updateVote', json={
                             "boomerIndex": loaded_json["boomerIndex"], "voteValue": isTrueVote}, headers=headers)
    print("update status" + str(response.status_code))
    response.raise_for_status

# HANDLER FUNCTIONS


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="Send me a message using /check, we'll see if its real or not!")


def verify(update, context):

    context.bot.send_message(
        chat_id=update.effective_chat.id, text="Loading links...")

    global loaded_json

    # Dummy dataset from API Call
    #loaded_json = {"googleLinks":["www.google.com.sg","www.yahoo.com.sg"],"percentageFakeNews":50.0,"trueCount":1,"falseCount":1,"totalSearchCount":2}

    # Configure and call API for reputable links given message
    loaded_json = getCheckFakeNewsResults(HOST, update)
    print(loaded_json)

    # Formatting and displaying links to user
    link_text = "Here are the reputable articles we have found:"
    links = loaded_json["googleLinks"]
    for i in range(len(links)):
        entry = "\n" + str(i+1) + ". {}".format(links[i])
        link_text += entry
    context.bot.send_message(chat_id=update.effective_chat.id, text=link_text)

    # Prompting for user vote
    '''
    totalSearchCount = loaded_json["totalSearchCount"]
    print(totalSearchCount)
    user_word = ""
    if int(totalSearchCount) == 1:
        user_word =  "user" 
    else: 
        user_word = "users"
    print(user_word)
    '''
    question = "{} users have checked for a similar message.".format(
        loaded_json["totalSearchCount"])
    question += "\nCurrent poll flags this message as {:.1f}% fake.".format(
        loaded_json["percentageFakeNews"])
    question += "\nFrom the given sources above, how likely is this message fake news?"

    # Configure keyboard for voting
    keyboard = [[InlineKeyboardButton(
        "Real", callback_data="1"), InlineKeyboardButton("Fake", callback_data="0")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    context.bot.send_message(
        chat_id=update.effective_chat.id, text=question, reply_markup=reply_markup)


def button(update, context):
    query = update.callback_query
    query.answer()

    global loaded_json
    trueCount = loaded_json["trueCount"]
    falseCount = loaded_json["falseCount"]

    isTrueVote = int(query.data)
    if (isTrueVote):
        trueCount += 1
    else:
        falseCount += 1
    total = trueCount + falseCount

    query.edit_message_text(text="Poll results: Real {:.1f}% | Fake {:.1f}%".format(
        getPercentage(trueCount, total), getPercentage(falseCount, total)))

    # Call API to update database with new user vote
    updateVote(HOST, isTrueVote)

# MAIN FUNCTION

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text, verify))
    dp.add_handler(CallbackQueryHandler(button))
    updater.start_polling()
    #updater.start_webhook(listen="0.0.0.0",port=PORT,url_path=TOKEN)
    #updater.bot.set_webhook("https://hidden-reef-68700.herokuapp.com/" + TOKEN)
    updater.idle()


if __name__ == '__main__':
    main()
