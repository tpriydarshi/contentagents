from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
print("Environment variables loaded")

# Initialize OpenAI client
try:
    client = OpenAI()
    print("OpenAI client initialized successfully")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")

def clarity_agent(user_input):
    """First agent to understand user needs and clarify content requirements"""
    print(f"\nClarity Agent processing input: '{user_input}'")
    try:
        clarity_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clarity agent for a biopharma content team. Your role is to understand and clarify the user's content needs."},
                {"role": "user", "content": user_input}
            ]
        )
        print("Clarity Agent completed successfully")
        return clarity_response.choices[0].message.content
    except Exception as e:
        print(f"Error in Clarity Agent: {e}")
        raise

def creator_agent(clarity_output):
    """Second agent to generate image content using DALL-E 3"""
    print("\nCreator Agent starting...")
    try:
        # First, create an optimized prompt for DALL-E 3
        prompt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a prompt engineer for DALL-E 3. Create a detailed, clear prompt that will generate a professional biopharma-related image."},
                {"role": "user", "content": f"Convert these requirements into a DALL-E 3 prompt: {clarity_output}"}
            ]
        )
        
        optimized_prompt = prompt_response.choices[0].message.content
        print(f"Generated DALL-E prompt: {optimized_prompt}")

        # Generate image using DALL-E 3
        image_response = client.images.generate(
            model="dall-e-3",
            prompt=optimized_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = image_response.data[0].url
        print("Image generated successfully")
        
        return {
            "image_url": image_url,
            "prompt_used": optimized_prompt
        }
    except Exception as e:
        print(f"Error in Creator Agent: {e}")
        raise

def copy_agent(clarity_output, creator_output):
    """Third agent to create a key message based on clarity and creator outputs"""
    print("\nCopy Agent starting...")
    try:
        copy_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a copy agent for biopharma content. Your role is to create one impactful key message that captures the essence of the content."},
                {"role": "user", "content": f"""Based on:
                Requirements: {clarity_output}
                Generated Image Prompt: {creator_output['prompt_used']}
                Image URL: {creator_output['image_url']}
                
                Please create one powerful key message that captures the core value proposition."""}
            ]
        )
        print("Copy Agent completed successfully")
        return copy_response.choices[0].message.content
    except Exception as e:
        print(f"Error in Copy Agent: {e}")
        raise

def assembler_agent(creator_output, clarity_output, copy_output, user_input):
    """Final agent to format and assemble the final content"""
    print("\nAssembler Agent starting...")
    try:
        assembler_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assembler agent for biopharma content. Your role is to organize and present the final content in a clear, professional format."},
                {"role": "user", "content": f"""Please organize all inputs into a cohesive final document:

Original Request: {user_input}
Clarified Requirements: {clarity_output}
Key Message: {copy_output}
Generated Image: {creator_output['image_url']}
Image Prompt Used: {creator_output['prompt_used']}

Please structure this into a professional document with clear sections, including where the image should be placed."""}
            ]
        )
        print("Assembler Agent completed successfully")
        return assembler_response.choices[0].message.content
    except Exception as e:
        print(f"Error in Assembler Agent: {e}")
        raise

def main():
    try:
        # Get user input
        user_input = input("What kind of biopharma content do you need help with? ")
        print("\nStarting content generation pipeline...")
        
        # Step 1: Clarity Agent
        clarity_output = clarity_agent(user_input)
        print("Clarity Agent Output:\n", clarity_output, "\n")
        
        # Step 2: Creator Agent (DALL-E 3)
        creator_output = creator_agent(clarity_output)
        print("Creator Agent Output:\n", f"Image URL: {creator_output['image_url']}\n")
        print(f"Prompt Used: {creator_output['prompt_used']}\n")
        
        # Step 3: Copy Agent
        copy_output = copy_agent(clarity_output, creator_output)
        print("Copy Agent Output (Key Message):\n", copy_output, "\n")
        
        # Step 4: Assembler Agent
        final_output = assembler_agent(creator_output, clarity_output, copy_output, user_input)
        print("Final Assembled Output:\n", final_output)
        
    except Exception as e:
        print(f"\nError in main execution: {e}")

if __name__ == "__main__":
    main() 