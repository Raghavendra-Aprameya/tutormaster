import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()


class TeachingAgent:
    """
    LangChain-based teaching agent that uses tool calling to route between
    comprehensive teaching and quick Q&A based on user intent.
    
    Hardcoded to: Subject=History, Class Level=10
    All responses come exclusively from the vector database (PDF document).
    """
    
    # Hardcoded constants
    SUBJECT = "History"
    CLASS_LEVEL = "10"
    CHAPTER = "A Brief History of India"  # Default chapter
    
    def __init__(self, subject: str = None, class_level: str = None, chapter: str = None, study_material_id: str = None):
        """
        Initialize the teaching agent with LangChain agent framework
        
        Args:
            subject: Subject name (e.g., "History")
            class_level: Class level/grade (e.g., "10")
            chapter: Chapter/title name (e.g., "A Brief History of India")
            study_material_id: Study material ID from PostgreSQL (used to filter embeddings)
        """
        self.subject = subject or self.SUBJECT
        self.class_level = class_level or self.CLASS_LEVEL
        self.chapter = chapter or self.CHAPTER
        self.study_material_id = study_material_id
        
        self.chat_history: List[Dict[str, str]] = []
        self.current_mode: Optional[str] = None  # "teaching" or "qa"
        self.teaching_context = {
            "subject": self.subject,
            "chapter": self.chapter,
            "class_level": self.class_level,
            "topics_covered": [],
            "current_topic": None,
            "last_teaching_point": "",
            "paused": False
        }
        self.loaded_chunks = []
        
        # Topic management
        self.topics_to_teach: List[str] = []  # List of all topics to teach
        self.current_topic: Optional[str] = None  # Current topic being taught
        self.next_topic: Optional[str] = None  # Next topic in the list
        self.current_topic_complete: bool = False  # Boolean indicating if current topic is done
        
        # Initialize LLM
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        self.llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=openai_endpoint,
            api_version="2024-12-01-preview",
            azure_deployment="gpt-4o-mini",
            temperature=0.5,
        )
        
        # Initialize tools (will be loaded lazily)
        self._teaching_tool_instance = None
        self._qa_tool_instance = None
        
        # Setup agent
        self._setup_agent()
    
    def _get_teaching_tool_instance(self):
        """Lazy load teaching tool"""
        if self._teaching_tool_instance is None:
            from interactive_tutor import InteractiveTutorTool
            self._teaching_tool_instance = InteractiveTutorTool(
                subject=self.subject,
                class_level=self.class_level,
                chapter=self.chapter,
                study_material_id=self.study_material_id
            )
        return self._teaching_tool_instance
    
    def _get_qa_tool_instance(self):
        """Lazy load Q&A tool"""
        if self._qa_tool_instance is None:
            from tutionmaster_faq import TutorLLMNode
            self._qa_tool_instance = TutorLLMNode()
        return self._qa_tool_instance
    
    def _setup_agent(self):
        """Setup LangChain agent with tools"""
        
        # Create tools using decorators and closures to access self
        @tool
        def start_teaching_session() -> str:
            """
            Start a comprehensive interactive teaching session on Indian History (Class 10).
            Use this when the student wants to learn about Indian history or requests teaching.
            
            This will teach: A Brief History of India (Class 10 History)
            All content comes from the uploaded PDF document.
            
            Returns:
                Initial teaching message introducing the chapter and first topic
            """
            return self._start_teaching_internal(self.subject, self.chapter, self.class_level)
        
        @tool
        def answer_specific_question(question: str) -> str:
            """
            Answer a specific factual question using ONLY content from the History PDF (Class 10).
            Use this for quick factual lookups like dates, definitions, names, or specific facts.
            Questions like 'What year...?', 'Who was...?', 'Define...', 'When did...?'
            
            IMPORTANT: Answers come exclusively from "A Brief History of India" PDF document.
            
            If in a teaching session, this will pause teaching, answer the question from PDF, 
            and automatically resume teaching.
            
            Args:
                question: The specific factual question to answer from the PDF
            
            Returns:
                Concise factual answer from PDF, followed by teaching continuation if in teaching mode
            """
            return self._answer_question_internal(question)
        
        # Store tools (only 2 tools now - continue_teaching handled by state management)
        self.tools = [start_teaching_session, answer_specific_question]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent educational assistant teaching Class 10 History from "A Brief History of India" PDF document.

CRITICAL: ALL answers MUST come from the vector database (PDF document). NEVER use general knowledge or make assumptions.

You have access to TWO tools. Analyze the user's intent and choose the appropriate tool:

1. start_teaching_session() - Use when:
   - User wants to STUDY or LEARN a topic comprehensively
   - User wants to understand a topic in depth, not just get a quick answer
   - User wants structured, comprehensive teaching about a subject
   - Current mode is None (no active session)
   - Takes NO parameters - automatically uses History Class 10
   - Examples: User wants to learn about "Mughal Empire", "Indian Independence Movement", "Ancient India"
   - The intent is to TEACH/STUDY a broad topic, not answer a specific question

2. answer_specific_question(question) - Use when:
   - User asks a SPECIFIC QUESTION about the document
   - User wants a quick, factual answer to a specific query
   - User asks about a particular fact, event, person, date, or detail
   - Examples: "Who wrote Bande Mataram?", "What year was the Battle of Plassey?", "When did Gandhi start the Salt March?"
   - Works in both teaching and non-teaching modes
   - If in teaching mode, this will PAUSE teaching, answer the question, and wait for user to say "continue" to resume

INTENT ANALYSIS:
- Understand the user's intent: Do they want to STUDY/LEARN a topic (teaching mode) or get a SPECIFIC ANSWER (FAQ mode)?
- Teaching mode: Comprehensive learning about a topic/subject
- FAQ mode: Quick answers to specific questions about facts/details in the document

STATE MANAGEMENT:
- If current_mode is "teaching" AND the message is a specific question, use answer_specific_question() 
  (it will PAUSE teaching, answer the question, and teaching will STAY PAUSED until user explicitly continues).
- If current_mode is "teaching" AND the message is NOT a specific question (general response, continuation), 
  DO NOT call any tool - the system will automatically continue teaching (state management handles this).
- If current_mode is "teaching" AND teaching is PAUSED:
  - If user asks another question → use answer_specific_question() (teaching stays paused)
  - If user wants to continue (says "continue", "go on", etc.) → DO NOT call any tool, system will automatically resume
- IMPORTANT: When teaching is paused, it stays paused until user explicitly requests to continue. Do not automatically resume.

DECISION GUIDELINES:
- Analyze the user's intent, not just keywords
- If user wants to study/learn a topic → start_teaching_session()
- If user asks a specific question → answer_specific_question()
- When in doubt, consider: Is this a request to learn about a topic, or a specific question?

Current Mode: {current_mode}
Teaching Context: {teaching_context}

TOPIC MANAGEMENT:
- When in teaching mode, focus on the current topic until it's comprehensively covered
- When a topic is complete, naturally transition to the next topic
- Use phrases like "Now let's move on to..." or "Next, we'll cover..." to signal topic transitions

IMPORTANT: Use ONLY ONE tool per message. Do not chain multiple tools. Return the tool result directly to the student.

Subject: History (Class 10)
Chapter: A Brief History of India

Always be natural and helpful, but strictly use only PDF content."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,  # Set to False to reduce noise, True for debugging
            handle_parsing_errors=True,
            max_iterations=10,  # Increased to allow more complex reasoning
            max_execution_time=30,  # 30 second timeout
            return_intermediate_steps=False
        )
    
    def process_message(self, user_message: str) -> str:
        """
        Process user message through LangChain agent with state management.
        If in teaching mode and message is not a specific question, automatically continue teaching.
        
        Args:
            user_message: User's input
            
        Returns:
            Agent's response
        """
        # Add to chat history
        self.chat_history.append({
            "role": "user",
            "content": user_message
        })
        
        # STATE MANAGEMENT: If in teaching mode and paused, check if user wants to continue
        if self.current_mode == "teaching" and self.teaching_context.get("paused", False):
            # Teaching is paused - check if user wants to continue
            message_lower = user_message.lower().strip()
            continue_indicators = ["continue", "go on", "resume", "keep going", "proceed", "next", "yes", "ok", "okay", "sure"]
            is_short_response = len(message_lower.split()) <= 3
            seems_like_continue = any(indicator in message_lower for indicator in continue_indicators) or (is_short_response and "?" not in user_message)
            
            if seems_like_continue:
                # User wants to continue - resume teaching from where we left off
                response = self._continue_teaching_internal(user_message)
                
                # Add response to history
                self.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                return response
            # If paused but user asks another question, let agent handle it (will call answer_specific_question, stays paused)
        
        # Convert chat history to LangChain format
        lc_history = []
        for msg in self.chat_history[:-1]:  # Exclude current message
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_history.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                lc_history.append(SystemMessage(content=msg["content"]))
        
        # Prepare context for agent
        context_summary = f"Mode: {self.current_mode or 'None'}"
        if self.current_mode == "teaching":
            context_summary += f", Chapter: {self.teaching_context.get('chapter', 'N/A')}"
            context_summary += f", Current Topic: {self.current_topic or 'N/A'}"
            if self.next_topic:
                context_summary += f", Next Topic: {self.next_topic}"
            context_summary += f", Topic Complete: {self.current_topic_complete}"
            context_summary += f", Topics Remaining: {len(self.topics_to_teach) - len(self.teaching_context.get('topics_covered', []))}"
            if self.teaching_context.get("paused", False):
                context_summary += f", Teaching is PAUSED - user can ask questions or say 'continue' to resume"
        
        # Track pause state before agent call
        was_paused_before = self.teaching_context.get("paused", False)
        
        try:
            # Invoke agent - let it decide based on intent
            result = self.agent_executor.invoke({
                "input": user_message,
                "chat_history": lc_history,
                "current_mode": self.current_mode or "None",
                "teaching_context": context_summary
            })
            
            response = result["output"]
            
            # Check pause state after agent call
            is_paused_after = self.teaching_context.get("paused", False)
            
            # If in teaching mode:
            if self.current_mode == "teaching":
                # If teaching became paused (wasn't paused before, is paused now)
                # → answer_specific_question was called, use its response (teaching is now paused)
                if not was_paused_before and is_paused_after:
                    # answer_specific_question was called, teaching is now paused
                    # Use the FAQ response (already in response variable)
                    pass
                # If teaching was already paused and is still paused
                # → user asked another question, answer_specific_question was called again
                # → Use the FAQ response (already in response variable)
                elif was_paused_before and is_paused_after:
                    # Still paused, another question was answered
                    # Use the FAQ response (already in response variable)
                    pass
                # If teaching is NOT paused (wasn't paused before, still not paused)
                # → agent didn't call answer_specific_question, so this was a continuation
                # → Continue teaching automatically
                elif not was_paused_before and not is_paused_after:
                    # Not paused, agent didn't call answer_specific_question
                    # This was a continuation message - continue teaching automatically
                    response = self._continue_teaching_internal(user_message)
                # If teaching was paused but is now unpaused
                # → This shouldn't happen from agent call, but if it does, continue teaching
                elif was_paused_before and not is_paused_after:
                    # Was paused, now unpaused - continue teaching
                    response = self._continue_teaching_internal(user_message)
            
            # Check if agent hit max iterations
            if "max iterations" in response.lower() or "stopped due to" in response.lower():
                # Fallback: try direct tool call based on context
                print("Agent hit max iterations, using fallback...")
                response = self._fallback_response(user_message)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Agent error: {error_msg}")
            
            # If max iterations error, use fallback
            if "max iterations" in error_msg.lower() or "max_execution_time" in error_msg.lower():
                response = self._fallback_response(user_message)
            else:
                # If in teaching mode and error, try to continue teaching
                if self.current_mode == "teaching":
                    response = self._continue_teaching_internal(user_message)
                else:
                    response = "I apologize, but I encountered an error. Could you please rephrase your question?"
        
        # Add response to history
        self.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _fallback_response(self, user_message: str) -> str:
        """
        Fallback method when agent hits max iterations.
        Simple fallback: if in teaching mode, continue teaching; otherwise default to FAQ.
        """
        # Simple fallback logic without keyword matching
        if self.current_mode == "teaching":
            # In teaching mode - continue teaching
            return self._continue_teaching_internal(user_message)
        else:
            # Not in teaching mode - default to FAQ (answer as question)
            # This is a safe default since most queries are questions
            return self._answer_question_internal(user_message)
    
    def _start_teaching_internal(self, subject: str, chapter: str, class_level: str) -> str:
        """Internal method to start teaching session"""
        self.current_mode = "teaching"
        self.teaching_context.update({
            "subject": subject,
            "chapter": chapter,
            "class_level": class_level,
            "topics_covered": [],
            "current_topic": None,
            "paused": False
        })
        
        # Add system message
        system_msg = f"Teaching session started: {subject} - {chapter} (Class {class_level})"
        self.chat_history.append({
            "role": "system",
            "content": system_msg
        })
        
        # Get teaching tool
        teaching_tool = self._get_teaching_tool_instance()
        
        # Extract topics
        topics = teaching_tool.extract_topics(subject, chapter, class_level)
        
        # Initialize topic management
        self.topics_to_teach = topics if topics else []
        if self.topics_to_teach:
            self.current_topic = self.topics_to_teach[0]
            self.next_topic = self.topics_to_teach[1] if len(self.topics_to_teach) > 1 else None
        else:
            self.current_topic = None
            self.next_topic = None
        self.current_topic_complete = False
        
        # Sync topic to teaching_context
        self._sync_topic_to_context()
        
        # Load chunks
        self.loaded_chunks = teaching_tool.load_chapter_chunks(subject, chapter, class_level)
        
        # Generate initial teaching
        initial_message = teaching_tool.generate_initial_teaching(
            subject, chapter, class_level, topics, self.loaded_chunks
        )
        
        return initial_message
    
    def _sync_topic_to_context(self):
        """Sync current_topic field with teaching_context"""
        self.teaching_context["current_topic"] = self.current_topic
    
    def _move_to_next_topic(self):
        """Handle transition to next topic when current topic is complete"""
        if not self.current_topic_complete:
            return
        
        # Move current topic to topics_covered
        if self.current_topic:
            self.teaching_context["topics_covered"].append(self.current_topic)
        
        # Move next topic to current topic
        if self.next_topic:
            self.current_topic = self.next_topic
            self._sync_topic_to_context()
            
            # Find index of next topic in the list
            try:
                current_index = self.topics_to_teach.index(self.current_topic)
                if current_index + 1 < len(self.topics_to_teach):
                    self.next_topic = self.topics_to_teach[current_index + 1]
                else:
                    self.next_topic = None
            except ValueError:
                self.next_topic = None
        else:
            # No more topics
            self.current_topic = None
            self._sync_topic_to_context()
        
        # Reset completion flag
        self.current_topic_complete = False
    
    def mark_topic_complete(self):
        """Manually mark current topic as complete (useful for external triggers)"""
        if self.current_mode == "teaching" and self.current_topic:
            self.current_topic_complete = True
            self._move_to_next_topic()
    
    def _check_topic_completion(self, response: str) -> bool:
        """
        Check if the teaching response indicates current topic is complete.
        Looks for signals like topic transitions, summaries, or explicit completion.
        """
        response_lower = response.lower()
        
        # Check for explicit completion signals
        completion_keywords = [
            "now let's move on",
            "next topic",
            "moving to",
            "let's now discuss",
            "now we'll cover",
            "next, we'll",
            "that covers",
            "we've covered",
            "now let's explore",
            "transitioning to",
            "moving forward to"
        ]
        
        # Check if response mentions next topic explicitly
        if self.next_topic:
            if self.next_topic.lower() in response_lower:
                return True
        
        # Check for completion keywords
        for keyword in completion_keywords:
            if keyword in response_lower:
                return True
        
        return False
    
    def _continue_teaching_internal(self, student_message: str) -> str:
        """
        Internal method to continue teaching.
        If teaching was paused, resumes from the last teaching point.
        Otherwise, continues normally.
        """
        if self.current_mode != "teaching":
            return "No active teaching session. Please start a teaching session first."
        
        teaching_tool = self._get_teaching_tool_instance()
        
        # Check if teaching was paused - if so, resume from where we left off
        if self.teaching_context.get("paused", False):
            # RESUME: Unpause and continue from last teaching point
            self.teaching_context["paused"] = False
        
        # Check if we need to move to next topic
        if self.current_topic_complete:
            self._move_to_next_topic()
        
            # Create resume prompt that references where we left off
            last_point = self.teaching_context.get("last_teaching_point", "")
            current_topic = self.current_topic or "the current topic"
            
            # Check if user explicitly wants to continue (keywords like "continue", "go on", etc.)
            continue_keywords = ["continue", "go on", "resume", "keep going", "proceed", "next"]
            is_explicit_continue = any(keyword in student_message.lower() for keyword in continue_keywords)
            
            if is_explicit_continue or not student_message.strip():
                # User wants to continue - resume from last point
                resume_prompt = f"""The student wants to continue learning. Resume teaching from where we left off using ONLY PDF content.

Last teaching point we covered:
{last_point}

Current topic: {current_topic}
{f'Next topic: {self.next_topic}' if self.next_topic else ''}

Continue teaching naturally from where we paused. Use a brief transition like "Now, continuing with..." or "Let's pick up where we left off..." and then continue teaching the current topic."""
            else:
                # User has a follow-up question/comment - incorporate it
                resume_prompt = f"""The student asked a question which was answered. Now they want to continue learning. 
Resume teaching from where we left off, incorporating their follow-up if relevant.

Last teaching point we covered:
{last_point}

Student's follow-up: {student_message}

Current topic: {current_topic}
{f'Next topic: {self.next_topic}' if self.next_topic else ''}

Continue teaching naturally, addressing their follow-up if relevant, then continue with the current topic."""
            
            response = teaching_tool.continue_teaching(
                chat_history=self.chat_history,
                chapter_chunks=self.loaded_chunks,
                user_message=resume_prompt
            )
            
            # Clear the last teaching point after resuming
            self.teaching_context["last_teaching_point"] = ""
        else:
            # Normal continuation (not resuming from pause)
            # Check if we need to move to next topic
            if self.current_topic_complete:
                self._move_to_next_topic()
        
        # Enhance user message with topic context for better teaching
        enhanced_message = student_message
        if self.current_topic:
            # Add context to help teaching tool stay focused on current topic
            if not any(keyword in student_message.lower() for keyword in ["next", "move on", "continue", "skip"]):
                enhanced_message = f"Continue teaching about {self.current_topic}. {student_message}"
        
        response = teaching_tool.continue_teaching(
            chat_history=self.chat_history,
            chapter_chunks=self.loaded_chunks,
            user_message=enhanced_message
        )
        
        # Check if current topic is complete based on response
        if self.current_topic and not self.current_topic_complete:
            if self._check_topic_completion(response):
                self.current_topic_complete = True
                # If topic is complete, move to next topic
                self._move_to_next_topic()
        
        return response
    
    def _answer_question_internal(self, question: str) -> str:
        """
        Internal method to answer specific questions.
        If in teaching mode, pauses teaching and answers the question.
        Teaching will resume when user explicitly requests to continue.
        """
        qa_tool = self._get_qa_tool_instance()
        
        # Always use hardcoded filter for History Class 10
        filter_meta = {
            "$and": [
                {"subject": {"$eq": self.subject}},
                {"chapter": {"$eq": self.chapter}},
                {"class_level": {"$eq": self.class_level}}
            ]
        }
        
        # If in teaching mode, implement pause-answer (but don't resume automatically)
        if self.current_mode == "teaching":
            # PAUSE: Save the current teaching point before answering
            self.teaching_context["paused"] = True
            
            # Save the last teaching point so we can resume from there
            if len(self.chat_history) >= 2:
                # Look for the last assistant message (teaching response)
                for msg in reversed(self.chat_history[:-1]):  # Exclude current user message
                    if msg["role"] == "assistant":
                        self.teaching_context["last_teaching_point"] = msg["content"]
                        break
            
            # If we couldn't find a teaching point, use a default
            if "last_teaching_point" not in self.teaching_context or not self.teaching_context["last_teaching_point"]:
                self.teaching_context["last_teaching_point"] = "We were discussing the current topic."
            
            # ANSWER - with strict filter
            qa_result = qa_tool.query_and_respond_with_history(
                query_text=question,
                chat_history=self.chat_history[:-1],
                filter_meta=filter_meta,
                k=6
            )
            
            answer = qa_result.get("response", "I couldn't find this information in our History textbook.")
            
            # Return answer with a note that teaching is paused
            # User needs to explicitly say "continue" to resume teaching
            return f"{answer}\n\n[Teaching paused. Say 'continue' or 'go on' to resume teaching from where we left off.]"
        
        else:
            # Not in teaching mode, just answer the question
            qa_result = qa_tool.query_and_respond_with_history(
                query_text=question,
                chat_history=self.chat_history[:-1],
                filter_meta=filter_meta,
                k=6
            )
            
            return qa_result.get("response", "I couldn't find this information in our History textbook.")


def run_agent():
    """
    Run the interactive teaching agent in command-line mode.
    """
    agent = TeachingAgent()
    
    print("=" * 80)
    print("Interactive Teaching Agent (LangChain-Powered)")
    print("=" * 80)
    print("\nWelcome! I'm your AI teaching assistant powered by LangChain.")
    print("\nI can:")
    print("  • Teach you comprehensive lessons on any chapter")
    print("  • Answer specific questions during teaching")
    print("  • Provide quick factual answers")
    print("\nExamples:")
    print("  - 'Teach me about ancient Indian history'")
    print("  - 'Tell me more about the Indus Valley'")
    print("  - 'What year was it discovered?'")
    print("\nType 'exit' to quit")
    print("=" * 80)
    
    while True:
        try:
            user_input = input("\n\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye! Happy learning!")
                break
            
            # Process through agent
            response = agent.process_message(user_input)
            
            print(f"\nAssistant: {response}")
            
            # Show mode and topic information
            if agent.current_mode:
                print(f"\n[Mode: {agent.current_mode}]")
                if agent.current_mode == "teaching":
                    if agent.current_topic:
                        print(f"[Current Topic: {agent.current_topic}]")
                    if agent.next_topic:
                        print(f"[Next Topic: {agent.next_topic}]")
                    if agent.topics_to_teach:
                        print(f"[Topics to Teach: {', '.join(agent.topics_to_teach)}]")
                    print(f"[Topic Complete: {agent.current_topic_complete}]")
                    print(f"[Topics Covered: {len(agent.teaching_context.get('topics_covered', []))}/{len(agent.topics_to_teach)}]")
                
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    run_agent()
