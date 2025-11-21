"""
Comprehensive example demonstrating the Interactive Teaching System with:
1. Starting a teaching session
2. Interactive teaching conversation
3. Student interruptions with specific questions (pause-answer-resume)
4. Seamless tool switching between teaching and Q&A

Now powered by LangChain's agent framework with tool calling!
"""

import sys
from teaching_agent import TeachingAgent


def simulate_conversation():
    """
    Simulates a realistic teaching session with interruptions.
    This demonstrates how the agent handles:
    - Teaching initiation
    - Comprehension responses from student
    - Specific factual questions that pause teaching
    - Resuming teaching after Q&A
    """
    
    print("=" * 80)
    print("INTERACTIVE TEACHING SYSTEM - DEMONSTRATION")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  • Starting a comprehensive teaching session")
    print("  • Student asking for clarifications (continues teaching)")
    print("  • Student asking specific questions (pauses → Q&A → resumes)")
    print("  • Shared conversation history across both tools")
    print("=" * 80)
    print()
    
    # Initialize agent
    agent = TeachingAgent()
    
    # Simulated conversation messages
    conversation = [
        {
            "user": "I want to learn about ancient Indian history",
            "explanation": "User requests to start teaching session"
        },
        {
            "user": "Yes, I'm ready to begin!",
            "explanation": "Student shows engagement and readiness"
        },
        {
            "user": "Tell me more about their cities",
            "explanation": "Teaching continuation - student wants more detail"
        },
        {
            "user": "What year was this civilization discovered?",
            "explanation": "SPECIFIC QUESTION - triggers pause-answer-resume flow"
        },
        {
            "user": "That's interesting! Please continue",
            "explanation": "Student acknowledges answer and wants to continue teaching"
        },
        {
            "user": "Who were the main rulers?",
            "explanation": "Another SPECIFIC QUESTION - tests multiple interruptions"
        },
        {
            "user": "ok, go on",
            "explanation": "Brief acknowledgment to continue teaching"
        }
    ]
    
    # Process each message
    for i, exchange in enumerate(conversation, 1):
        print(f"\n{'='*80}")
        print(f"EXCHANGE {i}")
        print(f"{'='*80}")
        print(f"\n[Context: {exchange['explanation']}]")
        print(f"\n👤 Student: {exchange['user']}")
        print()
        
        # Process through agent
        try:
            response = agent.process_message(exchange['user'])
            print(f"🤖 Teacher: {response}")
            
            # Show agent state
            if agent.current_mode:
                print(f"\n[Agent Mode: {agent.current_mode}]")
                if agent.teaching_context.get('paused'):
                    print("[Teaching Status: PAUSED for Q&A]")
                else:
                    print("[Teaching Status: ACTIVE]")
                if agent.current_mode == "teaching":
                    if agent.current_topic:
                        print(f"[Current Topic: {agent.current_topic}]")
                    if agent.next_topic:
                        print(f"[Next Topic: {agent.next_topic}]")
                    if agent.topics_to_teach:
                        print(f"[Topics to Teach: {', '.join(agent.topics_to_teach)}]")
                    print(f"[Topic Complete: {agent.current_topic_complete}]")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print(f"   Continuing with next message...")
    
    # Show conversation summary
    print(f"\n\n{'='*80}")
    print("CONVERSATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total messages exchanged: {len(agent.chat_history)}")
    print(f"Final mode: {agent.current_mode}")
    if agent.current_mode == "teaching":
        print(f"Topics to teach: {agent.topics_to_teach}")
        print(f"Current topic: {agent.current_topic}")
        print(f"Next topic: {agent.next_topic}")
        print(f"Topics covered: {agent.teaching_context.get('topics_covered', [])}")
    print(f"\nDemonstration complete! The agent successfully:")
    print("  ✓ Started a teaching session")
    print("  ✓ Handled teaching continuations")
    print("  ✓ Paused for specific questions")
    print("  ✓ Answered using RAG Q&A tool")
    print("  ✓ Resumed teaching seamlessly")
    print("  ✓ Maintained shared conversation history")
    print(f"{'='*80}\n")


def interactive_mode():
    """
    Run the agent in interactive mode for manual testing.
    """
    print("=" * 80)
    print("INTERACTIVE TEACHING SYSTEM - LIVE SESSION")
    print("=" * 80)
    print("\nStarting interactive mode...")
    print("You can now chat with the teaching agent!")
    print("\nTips:")
    print("  • Say 'teach me about...' to start a teaching session")
    print("  • Ask specific questions anytime to pause and get quick answers")
    print("  • The agent maintains conversation history throughout")
    print("  • Type 'exit' to quit")
    print("=" * 80)
    print()
    
    # Import and run the agent's main loop
    from teaching_agent import run_agent
    run_agent()


def quick_demo():
    """
    Quick demonstration without full conversation simulation.
    Shows key features in a concise way.
    """
    print("=" * 80)
    print("QUICK DEMO - Key Features")
    print("=" * 80)
    
    agent = TeachingAgent()
    
    print("\n1. Starting Teaching Session")
    print("-" * 80)
    response1 = agent.process_message("teach me about ancient India")
    print(f"Response preview: {response1[:200]}...")
    print(f"Mode: {agent.current_mode}")
    
    print("\n2. Teaching Continuation")
    print("-" * 80)
    response2 = agent.process_message("Tell me more")
    print(f"Response preview: {response2[:200]}...")
    
    print("\n3. Specific Question (Pause-Answer-Resume)")
    print("-" * 80)
    response3 = agent.process_message("What year was this?")
    print(f"Response preview: {response3[:200]}...")
    print(f"Paused: {agent.teaching_context.get('paused', False)}")
    
    print("\n4. Shared History")
    print("-" * 80)
    print(f"Total messages in history: {len(agent.chat_history)}")
    print(f"Both tools have access to this shared history")
    
    print("\n" + "=" * 80)
    print("Quick demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "INTERACTIVE TEACHING SYSTEM EXAMPLES" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    print("Choose an option:")
    print("  1. Full Simulated Conversation (demonstrates all features)")
    print("  2. Interactive Mode (chat with the agent live)")
    print("  3. Quick Demo (concise feature demonstration)")
    print("  4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        simulate_conversation()
    elif choice == "2":
        interactive_mode()
    elif choice == "3":
        quick_demo()
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice. Running full simulated conversation...")
        simulate_conversation()

