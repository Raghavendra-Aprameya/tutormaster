"""
Quick validation script to test that the Interactive Teaching System is properly set up.
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from teaching_agent import TeachingAgent
        print("  ✓ teaching_agent imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import teaching_agent: {e}")
        return False
    
    try:
        from interactive_tutor import InteractiveTutorTool
        print("  ✓ interactive_tutor imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import interactive_tutor: {e}")
        return False
    
    try:
        from tutionmaster_faq import TutorLLMNode
        print("  ✓ tutionmaster_faq imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import tutionmaster_faq: {e}")
        return False
    
    return True

def test_initialization():
    """Test that components can be initialized"""
    print("\nTesting initialization...")
    
    try:
        from teaching_agent import TeachingAgent
        agent = TeachingAgent()
        print("  ✓ TeachingAgent initialized")
        print(f"    - Chat history: {len(agent.chat_history)} messages")
        print(f"    - Current mode: {agent.current_mode}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        return False

def test_environment():
    """Test that required environment variables are set"""
    print("\nTesting environment variables...")
    
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT"
    ]
    
    all_set = True
    for var in required_vars:
        if os.getenv(var):
            print(f"  ✓ {var} is set")
        else:
            print(f"  ✗ {var} is NOT set")
            all_set = False
    
    return all_set

def test_chromadb():
    """Test ChromaDB connection"""
    print("\nTesting ChromaDB connection...")
    
    try:
        from interactive_tutor import InteractiveTutorTool
        tutor = InteractiveTutorTool()
        
        # Try to get available chapters
        chapters = tutor.get_available_chapters()
        print(f"  ✓ ChromaDB connected")
        print(f"    - Found {len(chapters)} chapter(s) in database")
        
        if chapters:
            print(f"    - Sample: {chapters[0]}")
        else:
            print("    - Warning: No chapters found. Have you run create_embeddings.py?")
        
        return True
    except Exception as e:
        print(f"  ✗ ChromaDB connection failed: {e}")
        return False

def test_basic_routing():
    """Test basic agent setup and tools"""
    print("\nTesting agent setup and tools...")
    
    try:
        from teaching_agent import TeachingAgent
        agent = TeachingAgent()
        
        # Test that agent executor is set up
        if hasattr(agent, 'agent_executor'):
            print(f"  ✓ Agent executor created")
        else:
            print(f"  ✗ Agent executor not found")
            return False
        
        # Test that tools are registered
        if hasattr(agent, 'tools') and len(agent.tools) > 0:
            print(f"  ✓ Tools registered: {len(agent.tools)} tools")
            for tool in agent.tools:
                print(f"    - {tool.name}")
        else:
            print(f"  ✗ No tools found")
            return False
        
        # Test LLM is initialized
        if hasattr(agent, 'llm'):
            print(f"  ✓ LLM initialized")
        else:
            print(f"  ✗ LLM not initialized")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Agent setup test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 80)
    print("INTERACTIVE TEACHING SYSTEM - SETUP VALIDATION")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Initialization", test_initialization()))
    results.append(("Environment", test_environment()))
    results.append(("ChromaDB", test_chromadb()))
    results.append(("Agent & Tools Setup", test_basic_routing()))
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python teaching_agent.py (for interactive mode)")
        print("  2. Run: python example_teaching_session.py (for demonstrations)")
        print("  3. See TEACHING_SYSTEM_README.md for full documentation")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  • Set environment variables in .env file")
        print("  • Run create_embeddings.py to populate ChromaDB")
        print("  • Install required packages: pip install -r requirements.txt")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

