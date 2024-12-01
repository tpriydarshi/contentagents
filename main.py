from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap

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
                {"role": "system", "content": "You are a clarity agent. Your role is to understand and enhance the user's image description while maintaining their exact requirements. Do not change the theme or core elements of their request."},
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
                {"role": "system", "content": "You are a prompt engineer for DALL-E 3. Your role is to enhance the given image description to create a more detailed and vivid prompt, while maintaining the original intent and core elements exactly as specified. Do not change the fundamental elements or theme of the request."},
                {"role": "user", "content": f"Enhance this image description into a detailed DALL-E prompt while keeping all original elements: {clarity_output}"}
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

def download_image(url):
    """Download image from URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise

def add_text_to_image(image, key_message, clarity_output):
    """Add text to image in a visually appealing way"""
    # Create a new image with extra space for text
    margin = 50
    spacing = 30
    text_height = 200  # Space for text below image
    
    # Create new white canvas
    new_img = Image.new('RGB', (image.width + 2*margin, image.height + text_height + 2*margin), 'white')
    
    # Paste original image
    new_img.paste(image, (margin, margin))
    
    # Prepare for text drawing
    draw = ImageDraw.Draw(new_img)
    
    try:
        # Try to load a nice font, fall back to default if not available
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add key message
    wrapped_key_message = textwrap.fill(key_message, width=50)
    draw.text((margin, image.height + margin + spacing), 
              wrapped_key_message, 
              font=font_large, 
              fill='black')
    
    return new_img

def save_final_content(image, output_dir="output"):
    """Save the final image to the output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename using timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/content_{timestamp}.png"
    
    # Save image
    image.save(filename)
    return filename

def assembler_agent(creator_output, clarity_output, copy_output, user_input):
    """Final agent to format and assemble the final content"""
    print("\nAssembler Agent starting...")
    try:
        # First, get the formatted text output
        assembler_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assembler agent for biopharma content. Your role is to organize and present the final content in a clear, professional format."},
                {"role": "user", "content": f"""Please organize all inputs into a cohesive final document:

Original Request: {user_input}
Clarified Requirements: {clarity_output}
Key Message: {copy_output}
Image Prompt Used: {creator_output['prompt_used']}

Please structure this into a professional document with clear sections."""}
            ]
        )
        
        # Download and process the image
        image = download_image(creator_output['image_url'])
        
        # Add text to image
        final_image = add_text_to_image(image, copy_output, clarity_output)
        
        # Save the final content
        output_file = save_final_content(final_image)
        
        print("Assembler Agent completed successfully")
        print(f"Final content saved to: {output_file}")
        
        return {
            'text_content': assembler_response.choices[0].message.content,
            'image_path': output_file
        }
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
        print("\nFinal Text Content:\n", final_output['text_content'])
        print(f"\nFinal image with text has been saved to: {final_output['image_path']}")
        
        # Try to open the generated image
        try:
            import subprocess
            subprocess.run(['open', final_output['image_path']])
            print("\nOpening the generated image...")
        except Exception as e:
            print(f"\nCouldn't automatically open the image: {e}")
            print("Please open the image manually from the output directory")
        
    except Exception as e:
        print(f"\nError in main execution: {e}")

if __name__ == "__main__":
    main()