import streamlit as st
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from llm import get_possible_objects
from vision import run_clip_on_objects, run_vlm
from utils import get_topk_paths_from_coord_data, get_topk_imgs_from_coord_data
from PIL import Image

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
    /* Sidebar styling */
    .status-sidebar {
        position: fixed;
        left: 0;
        top: 0;
        height: 100vh;
        width: 300px;
        background-color: #f0f2f6;
        padding: 2rem;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        transition: transform 0.3s ease-in-out;
        z-index: 1000;
    }
    .status-sidebar.hidden {
        transform: translateX(-100%);
    }
    .main-content {
        transition: margin-left 0.3s ease-in-out;
    }
    .main-content.sidebar-visible {
        margin-left: 300px;
    }
    /* Toggle button styling */
    .sidebar-toggle-btn {
        margin-bottom: 20px;
        padding: 8px 16px;
        border-radius: 20px;
        background-color: #3498db;
        color: white;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Global variable to track ROS2 initialization
if "ros_initialized" not in st.session_state:
    st.session_state.ros_initialized = False

class ROS2Interface(Node):
    def __init__(self):
        super().__init__('streamlit_interface')
        self.publisher_ = self.create_publisher(String, 'coordinate_data', 10)
        self.subscription = self.create_subscription(
            String,
            'ros_feedback',
            self.listener_callback,
            10
        )
        self.received_message = ''

    def listener_callback(self, msg):
        self.received_message = msg.data

    def publish(self, coord_data):        #
        msg = String()
        msg.data = json.dumps(coord_data)
        self.publisher_.publish(msg)


# Initialize ROS2 only once
if not st.session_state.ros_initialized:
    with st.spinner("Initializing ROS2 Connection..."):
        rclpy.init()
        st.session_state.ros_node = ROS2Interface()
        st.session_state.ros_initialized = True
        st.success("ROS2 Connection Established!")

ros_node = st.session_state.ros_node


# Load database collection only once
if "db_collection" not in st.session_state:
    with st.spinner("Loading swarm info..."):
        db_client = chromadb.HttpClient(host='localhost', port=8000)
        embedding_function = OpenCLIPEmbeddingFunction('ViT-B-16-SigLIP', 'webli', device='cuda')
        st.session_state.db_collection = db_client.get_collection(
            'test1', 
            embedding_function=embedding_function, 
            data_loader=ImageLoader()
        )

collection = st.session_state.db_collection


# Initialize session state for sidebar visibility
if 'show_sidebar' not in st.session_state:
    st.session_state.show_sidebar = True

# Create a container for the entire app
app_container = st.container()

# Status Sidebar
if st.session_state.show_sidebar:
    with st.sidebar:
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
        
        

# Main Content
with app_container:
    
    # Add toggle button at the top with better styling
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button('☰ Toggle Status Panel', 
                     key='sidebar_toggle',
                     use_container_width=True):
            st.session_state.show_sidebar = not st.session_state.show_sidebar
    
    
    st.title("🤖 Intelligent Swarm Robotics Command Center")
    st.markdown("""
    ### Welcome to the Robot Command Interface
    This system allows you to control swarm robots using natural language commands. 
    The robots will understand your instructions and navigate to the specified objects or locations.

    #### How it works:
    1. Enter your command (e.g., "Go to the nearest fire extinguisher")
    2. The system will identify relevant objects
    3. Robots will locate and navigate to the target
""")
    
    # Text input with placeholder
    prompt = st.text_input(
        "Enter your command",
        placeholder="Example: Find the nearest fire extinguisher and move towards it",
        help="Type a natural language command for the robots"
    )
    
    if st.button("🚀 Execute Command", type="primary") and prompt:
        st.markdown("### Processing Pipeline")

        try:
            # Step 1: Natural Language Processing
            with st.status("🧠 Understanding your command...", expanded=True) as status:
                # objects_json = get_possible_objects(prompt)
                # object_list = objects_json['possible_objects']
                object_list = ['bed', 'dustbin'] # for testing
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

            # After getting coord_data, display the detected objects
            with st.status("🖼️ Retrieving object images...", expanded=True) as status:
                top_matches = get_topk_imgs_from_coord_data(coord_data, k=4)  # Returns list of (object_name, image_path)

                # Create a container for images
                st.markdown("### Detected Objects")
                image_cols = st.columns(min(len(top_matches), 4))  # Show 3 images per row max

                for idx, (obj_name, image) in enumerate(top_matches):
                    col_idx = idx % 4  # Determine which column to put the image in
                    with image_cols[col_idx]:
                        try:
                            st.image(image, caption=f"{obj_name}", use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not load image for {obj_name}")

                status.update(label="✅ Object images retrieved!", state="complete")

            # Step 4: Robot Command Execution
            with st.status("🤖 Sending commands to robots...", expanded=True) as status:
                ros_node.publish(coord_data)
                status.update(label="✅ Commands sent to robots!", state="complete")

            st.success("🎉 Command successfully processed and sent to robots!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please try again or contact system administrator if the problem persists.")


st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <small>Intelligent Swarm Robotics System v1.0 | For assistance, contact support</small>
    </div>
    """, unsafe_allow_html=True)