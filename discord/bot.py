import discord
import os
import requests
from dotenv import load_dotenv

load_dotenv()


class MyBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.api_base_url = os.getenv("API_BASE_URL")
        if not self.api_base_url:
            raise ValueError("API_BASE_URL not set in environment")

        self.user_conversations = {}

    async def on_ready(self):
        print(f"Logged in as {self.user}\n")

    async def on_message(self, message):
        if message.author == self.user:
            return

        content = message.content
        if self.user in message.mentions:
            content = content.replace(f"<@{self.user.id}>", "").strip()
        
        if content.startswith("!"):
            command = content[1:].split()[0].lower()
            if command in ["memory", "help"]:
                await self.on_command(message, command)
                return
            else:
                return

        if not message.content.startswith("!") and self.user not in message.mentions:
            return

        user_id = str(message.author.id)

        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = f"{user_id}_{message.channel.id}"

        conversation_id = self.user_conversations[user_id]

        content = message.content
        if self.user in message.mentions:
            content = content.replace(f"<@{self.user.id}>", "").strip()
        if content.startswith("!"):
            content = content[1:].strip()

        if not content:
            return

        async with message.channel.typing():
            try:
                response = requests.post(
                    f"{self.api_base_url}/chat",
                    json={
                        "user_id": user_id,
                        "message_content": content,
                        "conversation_id": conversation_id,
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    bot_response = data.get(
                        "response",
                        "Hmm… I didn’t get that. Mind saying it another way?",
                    )

                    if len(bot_response) > 2000:
                        chunks = [
                            bot_response[i : i + 2000]
                            for i in range(0, len(bot_response), 2000)
                        ]
                        for chunk in chunks:
                            await message.reply(chunk)
                    else:
                        await message.reply(bot_response)
                else:
                    error_msg = f"Sorry, I encountered an error (Status: {response.status_code})"
                    await message.reply(error_msg)

            except requests.exceptions.Timeout:
                await message.reply("Sorry, request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                await message.reply(
                    "Sorry, I can't connect to my brain right now. Please try again later."
                )
            except Exception as e:
                print(f"Error processing message: {e}")
                await message.reply("Sorry, something went wrong. Please try again.")

    async def on_command(self, message, command):
        user_id = str(message.author.id)

        if command == "memory":
            try:
                response = requests.get(
                    f"{self.api_base_url}/memory/{user_id}", 
                    timeout=60,
                    headers={'Connection': 'keep-alive'}
                )
                if response.status_code == 200:
                    data = response.json()
                    summary = data.get("summary", "No memory data available.")
                    memory_response = f"**Memory Summary:**\n{summary}"

                    if len(memory_response) > 2000:
                        chunks = [
                            memory_response[i : i + 2000]
                            for i in range(0, len(memory_response), 2000)
                        ]
                        for chunk in chunks:
                            await message.reply(chunk)
                    else:
                        await message.reply(memory_response)
                else:
                    print(f"API Error: {response.status_code} - {response.text}")  
                    await message.reply("Couldn't retrieve memory summary.")
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error: {e}")  
                await message.reply("Can't connect to memory server.")
            except requests.exceptions.Timeout as e:
                print(f"Timeout error: {e}")  
                await message.reply("Memory request timed out.")
            except Exception as e:
                print(f"Exception in memory command: {type(e).__name__}: {e}")  
                await message.reply("Error getting memory summary.")

        elif command == "help":
            help_text = """
            **Bot Commands:**
            • `!help` - Show this help message
            • `!memory` - Show your memory summary
            • Just message me or mention me to chat!
            """
            await message.reply(help_text)


def run_bot():
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN not set in environment")

    bot = MyBot()
    bot.run(token)


if __name__ == "__main__":
    run_bot()
