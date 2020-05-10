import re
import string
import os
import requests
import json
from telegram.ext import *
from telegram import Bot,InlineKeyboardButton,InlineKeyboardMarkup,ReplyKeyboardRemove

TOKEN="1239297361:AAHCoR_LZycidG4mxbjfS6kEs7P0KZUUcQo"
#PORT = int(os.environ.get('PORT', '8443'))
bot = Bot(token = TOKEN)
loaded_json = {}
VERIFY = range(1)

#TEST FUNCTIONS

def get_url():
    contents = requests.get('https://random.dog/woof.json').json()
    url = contents['url']
    return url

def pic(update,context):
    url = get_url()
    chat_id = update.message.chat_id
    bot.send_photo(chat_id=chat_id, photo=url)

#ACTUAL FUNCTIONS

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Send me a message using /check, we'll see if its real or not!")
    r = requests.get('http://0.0.0.0:8010/loadBoomerData')
    print("load status" + str(r.status_code))
    r.raise_for_status

def check(update,context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="What message would you like to verify?")
    return VERIFY

def verify(update,context):
    keyboard = [[InlineKeyboardButton("Real",callback_data="1"),InlineKeyboardButton("Fake",callback_data="0")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    #First API Call
    global loaded_json
    #loaded_json = {"googleLinks":["www.google.com.sg","www.yahoo.com.sg"],"percentageFakeNews":50.0,"trueCount":1,"falseCount":1}
    headers = {'Content-type': 'application/json', 'Accept': '*/*'}
    r = requests.post('http://0.0.0.0:8010/checkFakeNews', json={"boomerText": update.message.text},headers=headers)
    context.bot.send_message(chat_id=update.effective_chat.id, text="Loading links...")
    loaded_json = r.json()
    print(loaded_json)

    link_text = "Here are the reputable articles we have found:"
    links = loaded_json["googleLinks"]
    for i in range(len(links)):
        entry = "\n" + str(i+1) + ". {}".format(links[i])
        link_text += entry
    context.bot.send_message(chat_id=update.effective_chat.id, text=link_text)

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
    question = "{} users have checked for a similar message.".format(loaded_json["totalSearchCount"])
    question += "\nCurrent poll flags this message as {:.1f}%\\ fake.".format(loaded_json["percentageFakeNews"])
    question += "\nFrom the given sources above, how likely is this message fake news?"
    
    context.bot.send_message(chat_id=update.effective_chat.id, text=question,reply_markup=reply_markup)
    
    return ConversationHandler.END
    
def getPercentage(number, total):
    return number/total * 100.0

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

    query.edit_message_text(text="Poll results: Real {:.1f}% | Fake {:.1f}%".format(getPercentage(trueCount,total),getPercentage(falseCount,total)))
    headers = {'Content-type': 'application/json', 'Accept': '*/*'}
    r = requests.post('http://0.0.0.0:8010/updateVote', json={"boomerIndex":loaded_json["boomerIndex"],"voteValue":isTrueVote},headers=headers)
    print("update status" + str(r.status_code))
    r.raise_for_status
    requests.post('http://0.0.0.0:8010/saveBoomerData',headers=headers)
    print("load status" + str(r.status_code))
    r.raise_for_status

    return ConversationHandler.END

def cancel(update, context):
    update.message.reply_text('Bye! Thank you for using FakeNewsBuster.',reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points = [CommandHandler('check',check)],
        states={
            VERIFY: [MessageHandler(Filters.text,verify)]
        },
        fallbacks=[CommandHandler('cancel',cancel)]
    )

    dp.add_handler(CommandHandler('start', start))
    #dp.add_handler(CommandHandler('pic', pic))
    dp.add_handler(conv_handler)
    dp.add_handler(CallbackQueryHandler(button))
    updater.start_polling()
    #updater.start_webhook(listen="0.0.0.0",port=PORT,url_path=TOKEN)
    #updater.bot.set_webhook("https://hidden-reef-68700.herokuapp.com/" + TOKEN)
    updater.idle()
    
if __name__ == '__main__':
    main()