import discord
import os
import httpx
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


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
        logger.info(f"Logged in as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return

        content = message.content
        logger.debug(f"Received message from {message.author.id}: {content[:50]}...")
        
        if self.user in message.mentions:
            content = content.replace(f"<@{self.user.id}>", "").strip()
            logger.debug(f"Bot mentioned, cleaned content: {content[:50]}...")
        
        if content.startswith("!"):
            command = content[1:].split()[0].lower()
            if command == "help":
                await self.on_command(message, command)
                return
            else:
                return

        if not message.content.startswith("!") and self.user not in message.mentions:
            return

        user_id = str(message.author.id)
        logger.debug(f"Processing message for user_id={user_id}")

        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = f"{user_id}_{message.channel.id}"
            logger.debug(f"Created new conversation for user_id={user_id}")

        conversation_id = self.user_conversations[user_id]

        content = message.content
        if self.user in message.mentions:
            content = content.replace(f"<@{self.user.id}>", "").strip()
        if content.startswith("!"):
            content = content[1:].strip()

        if not content:
            logger.debug("Empty content after cleaning, skipping")
            return

        logger.debug(f"Sending API request for user_id={user_id}, content_length={len(content)}")
        async with message.channel.typing():
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    logger.debug(f"POST to {self.api_base_url}/chat")
                    response = await client.post(
                        f"{self.api_base_url}/chat",
                        json={
                            "user_id": user_id,
                            "message_content": content,
                            "conversation_id": conversation_id,
                        },
                    )

                if response.status_code == 200:
                    data = response.json()
                    bot_response = data.get(
                        "response",
                        "Hmm… I didn't get that. Mind saying it another way?",
                    )
                    logger.debug(f"API response received, length={len(bot_response)}")

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

            except httpx.TimeoutException:
                logger.warning(f"Request timed out for user_id={user_id}")
                await message.reply("Sorry, request timed out. Please try again.")
            except httpx.ConnectError:
                logger.error(f"Cannot connect to API at {self.api_base_url}")
                await message.reply(
                    "Sorry, I can't connect to my brain right now. Please try again later."
                )
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await message.reply("Sorry, something went wrong. Please try again.")

    async def on_command(self, message, command):
        if command == "help":
            help_text = """
            **Bot Commands:**
            • `!help` - Show this help message
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
