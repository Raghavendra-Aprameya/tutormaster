"""
Example WebSocket client for Interactive Teaching Agent
Demonstrates how to connect and interact with the teaching agent via WebSocket
"""

import asyncio
import websockets
import json
import sys


async def interactive_client():
    """Interactive WebSocket client for testing the teaching agent"""
    
    # Generate unique client ID
    import random
    client_id = f"python_client_{random.randint(1000, 9999)}"
    uri = f"ws://localhost:8000/ws/{client_id}"
    
    print("=" * 80)
    print("Interactive Teaching Agent - WebSocket Client")
    print("=" * 80)
    print(f"Client ID: {client_id}")
    print(f"Connecting to: {uri}")
    print("=" * 80)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected to teaching agent!")
            
            # Receive welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            print(f"\nServer: {welcome_data.get('message', 'Connected')}")
            print("\nYou can now chat with the teaching agent.")
            print("Type 'exit' or 'quit' to disconnect.\n")
            print("=" * 80)
            
            # Start async tasks for sending and receiving
            async def receive_messages():
                """Continuously receive messages from server"""
                try:
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if data['type'] == 'response':
                            print(f"\n{'='*80}")
                            print(f"🤖 Teacher:")
                            print(f"{'='*80}")
                            print(data['response'])
                            
                            if data.get('mode'):
                                print(f"\n[Mode: {data['mode']}]", end="")
                            if data.get('current_topic'):
                                print(f" [Topic: {data['current_topic']}]", end="")
                            print("\n" + "=" * 80)
                            
                        elif data['type'] == 'error':
                            print(f"\n❌ Error: {data['message']}")
                            
                        elif data['type'] == 'system':
                            print(f"\nℹ️  System: {data['message']}")
                        
                        # Prompt for next input
                        print("\nYou: ", end="", flush=True)
                        
                except websockets.exceptions.ConnectionClosed:
                    print("\n\nConnection closed by server.")
                except Exception as e:
                    print(f"\nError receiving message: {e}")
            
            async def send_messages():
                """Continuously send messages to server"""
                try:
                    while True:
                        # Read user input
                        user_input = await asyncio.get_event_loop().run_in_executor(
                            None, input, "\nYou: "
                        )
                        
                        user_input = user_input.strip()
                        
                        if not user_input:
                            continue
                        
                        if user_input.lower() in ['exit', 'quit', 'bye']:
                            print("Disconnecting...")
                            break
                        
                        # Send message to server
                        message = {
                            "type": "message",
                            "content": user_input
                        }
                        await websocket.send(json.dumps(message))
                        
                except Exception as e:
                    print(f"\nError sending message: {e}")
            
            # Run both tasks concurrently
            await asyncio.gather(
                receive_messages(),
                send_messages()
            )
            
    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
        print("\nMake sure the server is running:")
        print("  python3 app.py")
    except ConnectionRefusedError:
        print("\n❌ Connection refused. Is the server running?")
        print("\nStart the server with:")
        print("  python3 app.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


async def automated_demo():
    """Automated demo showing various interactions"""
    
    client_id = "demo_client"
    uri = f"ws://localhost:8000/ws/{client_id}"
    
    print("=" * 80)
    print("Automated Demo - Teaching Agent WebSocket Client")
    print("=" * 80)
    
    test_messages = [
        "teach me about ancient India",
        "tell me more about the Indus Valley",
        "what year was this civilization discovered?",
        "continue teaching",
        "who were the main rulers?"
    ]
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected!\n")
            
            # Receive welcome
            welcome = await websocket.recv()
            print(f"Received: {welcome}\n")
            
            for i, message_text in enumerate(test_messages, 1):
                print(f"\n{'='*80}")
                print(f"Demo Step {i}/{len(test_messages)}")
                print(f"{'='*80}")
                print(f"👤 User: {message_text}")
                
                # Send message
                await websocket.send(json.dumps({
                    "type": "message",
                    "content": message_text
                }))
                
                # Receive response
                response = await websocket.recv()
                data = json.loads(response)
                
                if data['type'] == 'response':
                    print(f"\n🤖 Teacher:")
                    print(data['response'][:300] + "...")  # Truncate for demo
                    print(f"\nMode: {data.get('mode', 'N/A')}")
                else:
                    print(f"\nReceived: {data}")
                
                # Wait between messages
                await asyncio.sleep(2)
            
            print(f"\n{'='*80}")
            print("Demo completed successfully!")
            print(f"{'='*80}")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        print("\nRunning automated demo...\n")
        asyncio.run(automated_demo())
    else:
        print("\nStarting interactive client...\n")
        asyncio.run(interactive_client())


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Interactive Teaching Agent - WebSocket Client                       ║
║                                                                              ║
║  Usage:                                                                      ║
║    python3 websocket_client_example.py        # Interactive mode            ║
║    python3 websocket_client_example.py demo   # Automated demo              ║
║                                                                              ║
║  Make sure the server is running first:                                     ║
║    python3 app.py                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")

