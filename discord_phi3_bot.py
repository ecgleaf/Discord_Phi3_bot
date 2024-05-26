import discord
from discord.ext import commands
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load the Phi-3 mini model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model and tokenizer loaded.")

# Create an instance of a bot with the necessary intents
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
bot = commands.Bot(command_prefix='!', intents=intents)

executor = ThreadPoolExecutor(max_workers=4)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

def blocking_generate_response(question):
    try:
        print(f"Generating response for question: {question}")
        # Format the question in Q/A style
        formatted_question = f"Q: {question}\nA:"

        # Encode the input
        input_ids = tokenizer.encode(formatted_question, return_tensors='pt').to(device)
        
        # Generate the response
        generation_args = {
            "max_new_tokens": 100,
            "do_sample": True,  # Enable sampling for temperature and top_p settings to take effect
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }
        
        print(f"Input IDs: {input_ids}")
        output_ids = model.generate(input_ids, **generation_args)
        print(f"Output IDs: {output_ids}")

        # Decode the generated tokens
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the part of the response after "A:"
        if "A:" in response:
            response = response.split("A:")[1].strip()
        else:
            response = response.strip()
        
        print(f"Generated response: {response}")
        return response
    except Exception as e:
        print(f"Error during response generation: {e}")
        return "There was an error generating the response."

@bot.command(name='ask')
async def ask(ctx, *, question: str):
    print(f"Received question: {question}")
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, blocking_generate_response, question)
    except asyncio.TimeoutError:
        response = "Sorry, the response generation timed out."
    except Exception as e:
        print(f"Error in ask command: {e}")
        response = "An error occurred while generating the response."
    # Send the response back to the Discord channel
    await ctx.send(response)

# Replace 'YOUR_DISCORD_BOT_TOKEN' with your actual bot token
bot.run('YOUR_DISCORD_BOT_TOKEN')
