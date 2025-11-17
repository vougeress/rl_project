"""
System initialization script
"""
import asyncio
from api.core.learning_manager import learning_manager, initialize_learning_system

async def initialize_all_services():
    """Initialize all system services"""
    print("🚀 Initializing E-Commerce Recommendation System...")
    
    try:
        # Initialize learning system
        await initialize_learning_system()
        
        if learning_manager.is_ready:
            print("✅ All services initialized successfully!")
            return True
        else:
            print("❌ Learning system initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

# Для запуска при старте приложения
if __name__ == "__main__":
    asyncio.run(initialize_all_services())