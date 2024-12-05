import streamlit as st
from dotenv import load_dotenv
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from llm import get_possible_objects
from vision import run_clip_on_objects, run_vlm
from icecream import ic
import torch

torch.set_grad_enabled(False)

# Styling and Layout
st.set_page_config(
    page_title="Intelligent Swarm Robotics Interface",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stProgress .st-bo {
        background-color: #3498db;
    }
    .success-message {
        color: #2ecc71;
        font-weight: bold;
    }
    .processing-message {
        color: #3498db;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# Global variable to track ROS2 initialization
if "ros_initialized" not in st.session_state:
    st.session_state.ros_initialized = False

class ROS2Interface(Node):
    # [Previous ROS2Interface implementation remains the same]
    ...

# Initialize ROS2 only once
if not st.session_state.ros_initialized:
    with st.spinner("Initializing ROS2 Connection..."):
        rclpy.init()
        st.session_state.ros_node = ROS2Interface()
        st.session_state.ros_initialized = True
        st.success("ROS2 Connection Established!")

ros_node = st.session_state.ros_node

def load_db_collection():
    with st.spinner("Loading Database..."):
        db_client = chromadb.PersistentClient('/home/user1/s_ws/.chromadb_cache')
        embedding_function = OpenCLIPEmbeddingFunction('ViT-B-16-SigLIP', 'webli', device='cuda')
        collection = db_client.get_collection('test1', embedding_function=embedding_function, data_loader=ImageLoader())
        return collection

# Main UI
st.title("🤖 Intelligent Swarm Robotics Command Center")

st.markdown("""
    ### Welcome to the Robot Command Interface
    This system allows you to control swarm robots using natural language commands. 
    The robots will understand your instructions and navigate to the specified objects or locations.

    #### How it works:
    1. Enter your command (e.g., "Go to the nearest trash can")
    2. The system will identify relevant objects
    3. Robots will locate and navigate to the target
""")

# Initialize database
collection = load_db_collection()

# Text input with placeholder
prompt = st.text_input(
    "Enter your command",
    placeholder="Example: Find the nearest garbage bin and move towards it",
    help="Type a natural language command for the robots"
)

# Create two columns for main content and status
main_col, status_col = st.columns([2, 1])

with main_col:
    if st.button("🚀 Execute Command", type="primary") and prompt:
        st.markdown("### Processing Pipeline")

        try:
            # Step 1: Natural Language Processing
            with st.status("🧠 Understanding your command...", expanded=True) as status:
                objects_json = get_possible_objects(prompt)
                object_list = objects_json['possible_objects']
                st.write("Identified Objects:", ", ".join(object_list))
                status.update(label="✅ Command understood!", state="complete")

            # Step 2: Object Detection
            with st.status("🔍 Locating objects in environment...", expanded=True) as status:
                obj_path = run_clip_on_objects(object_list, collection)
                st.write("Located object at:", obj_path)
                status.update(label="✅ Objects located!", state="complete")

            # Step 3: Path Planning
            with st.status("📍 Calculating robot coordinates...", expanded=True) as status:
                coord_data = run_vlm(obj_path)
                st.write("Navigation coordinates:", coord_data)
                status.update(label="✅ Path planned!", state="complete")

            # Step 4: Robot Command Execution
            with st.status("🤖 Sending commands to robots...", expanded=True) as status:
                ros_node.publish(coord_data)
                status.update(label="✅ Commands sent to robots!", state="complete")

            st.success("🎉 Command successfully processed and sent to robots!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please try again or contact system administrator if the problem persists.")

with status_col:
    st.markdown("### 🤖 Robot Status")

    # Robot Feedback Section
    st.markdown("#### Live Feedback")
    if st.button("📡 Check Swarm Status"):
        with st.spinner("Receiving swarm feedback..."):
            rclpy.spin_once(ros_node, timeout_sec=0.1)
            if ros_node.received_message:
                st.success(f"📨 Latest Update: {ros_node.received_message}")
            else:
                st.info("⏳ No new update from swarm")

    # System Status
    st.markdown("#### System Status")
    status_indicator = "🟢" if st.session_state.ros_initialized else "🔴"
    st.markdown(f"ROS2 Connection: {status_indicator}")

    if st.button("🔄 Reset Connection"):
        st.session_state.ros_initialized = False
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <small>Intelligent Swarm Robotics System v1.0 | For assistance, contact support</small>
    </div>
    """, unsafe_allow_html=True)