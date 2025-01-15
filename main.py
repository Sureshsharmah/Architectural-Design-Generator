import openai
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Set OpenAI API key (replace with your OpenAI API key)
openai.api_key = "your_open_ai_key"

# Function to generate detailed architectural plans using OpenAI GPT model
def generate_architectural_plans():
    prompt = """
    You are an architect with knowledge of zoning laws, property constraints, and room design principles.
    Generate 5 distinct and alternative architectural plans for a residential property. Each plan should include:
    1. An overview of the design.
    2. Room layout details, such as the number and types of rooms (bedrooms, bathrooms, kitchen, etc.).
    3. Room sizes and their placement (how rooms are organized and where they are positioned in the house).
    4. Required materials and structural considerations.
    5. Any special design features or considerations (e.g., setbacks, natural light, ventilation, accessibility).
    6. A description of how the plan will work for the intended property.
    
    Provide these details in a clear, detailed format for each of the 5 plans.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an architect with knowledge of zoning laws, property constraints, and room design principles."},
                  {"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Function to generate different plots
def generate_plots():
    # Create random data for plotting
    np.random.seed(42)
    data = np.random.rand(10)
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    # Create Bar Plot
    fig1, ax1 = plt.subplots()
    ax1.bar(categories, data, color='skyblue')
    ax1.set_title("Bar Plot")
    ax1.set_xlabel("Categories")
    ax1.set_ylabel("Values")

    # Create Line Plot
    fig2, ax2 = plt.subplots()
    ax2.plot(categories, data, marker='o', color='green')
    ax2.set_title("Line Plot")
    ax2.set_xlabel("Categories")
    ax2.set_ylabel("Values")

    # Create Pie Chart
    fig3, ax3 = plt.subplots()
    ax3.pie(data, labels=categories, autopct='%1.1f%%', startangle=90)
    ax3.set_title("Pie Chart")

    # Create Histogram
    fig4, ax4 = plt.subplots()
    ax4.hist(data, bins=5, color='purple', edgecolor='black')
    ax4.set_title("Histogram")
    ax4.set_xlabel("Values")
    ax4.set_ylabel("Frequency")

    # Create Scatter Plot
    fig5, ax5 = plt.subplots()
    ax5.scatter(categories, data, color='red')
    ax5.set_title("Scatter Plot")
    ax5.set_xlabel("Categories")
    ax5.set_ylabel("Values")

    return fig1, fig2, fig3, fig4, fig5

# Streamlit UI
def main():
    # Streamlit page configuration
    st.set_page_config(page_title="Architectural Design Generator", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

    # App title and description
    st.title("🏡 Architectural Design Generator")
    st.markdown("Use this tool to generate *detailed and interactive architectural plans* for a residential property.")

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        section = st.radio("Choose a section:", ["Generate Plans", "View Plan Images", "Project Overview", "Model"])

    # Generate Plans section
    if section == "Generate Plans":
        st.header("🛠 Generate Architectural Plans")
        st.markdown("Click the button below to generate *5 unique architectural plans*:")
        if st.button("Generate Plans"):
            with st.spinner("Generating plans..."):
                plans = generate_architectural_plans()
                st.success("Architectural plans generated successfully!")
                st.markdown(plans)

    # View Plan Images section
    elif section == "View Plan Images":
        st.header("📷 View Plan Images")
        st.markdown("Explore visual representations of architectural plans by clicking the links below:")

        # Local paths to images
        plan_images = [
            {"title": "Plane 1", "path": "images/plane1.jpg"},
            {"title": "Plane 2", "path": "images/plane2.jpg"},
            {"title": "Plane 3", "path": "images/plane3.jpg"},
            {"title": "Plane 4", "path": "images/plane4.jpg"},
            {"title": "Plane 5", "path": "images/plane5.jpg"}
        ]

        # Display clickable links for each plan
        for plan in plan_images:
            if os.path.exists(plan["path"]):
                st.markdown(f"[{plan['title']}]({plan['path']})")
                if st.button(f"Show {plan['title']}"):
                    img = Image.open(plan["path"])
                    st.image(img, caption=plan['title'], use_container_width=True)
            else:
                st.error(f"Image not found: {plan['path']}")

    # Project Overview section
    elif section == "Project Overview":
        st.header("📋 Project Overview")
        st.markdown("""
        *Assignment Title*: Property-Driven Architectural Design Generator

        *Objective*:  
        The primary goal of this project is to develop an AI/ML-based system that can generate multiple alternative architectural plans for a given property while adhering to zoning and functional constraints. The system will:
        - Fetch and process property-specific details such as lot area, zoning rules, setbacks, and permissible Floor Area Ratio (FAR).
        - Propose diverse architectural plans that align with specified constraints, including room dimensions, MEP (mechanical, electrical, plumbing) layouts, and zoning rules.
        - Provide a visualization of these proposed designs in an easily understandable format.

        *Scope of Work*:  
        - *Data Acquisition*:
          - Simulate fetching property details from an API or database to ensure realistic inputs.
          - Details to Fetch:
            - Lot dimensions (width, length, area).
            - Zoning codes (e.g., residential, commercial).
            - Setback requirements (minimum distance from property boundaries).
            - Permissible building height and FAR.
            - Required rooms and amenities (e.g., bedrooms, bathrooms, kitchens).
            - Optional: Provide a mock dataset or create a simulated API to replicate real-world scenarios.

        - *Design Alternatives Generator*:
          - Leverage generative techniques to create multiple architectural designs.
          - Use machine learning models such as GANs, VAEs, or rule-based algorithms for plan generation.
          - Integrate zoning and functional constraints directly into the generation process to ensure compliance with regulatory standards.
          - Output: At least 5 diverse architectural plans per property input.

        - *Validation*:
          - Ensure compliance of generated plans with zoning laws and functional requirements.
          - Implement algorithms to verify adherence to constraints like setbacks, FAR, and minimum/maximum room sizes.
          - Automate detection of potential violations in MEP layouts.

        - *Visualization*:
          - Represent generated designs in a clear and intuitive manner.
          - Tasks:
            - Develop a simple interface or script to produce 2D sketches or schematic layouts of the generated plans.
            - Use visualization libraries like Matplotlib, Plotly, or CAD tools for rendering.

        *Deliverables*:  
        - Modular and well-documented code.
        - A complete pipeline consisting of:
          - Property data fetching.
          - Architectural plan generation.
          - Validation against constraints.
          - Visualization.
        - Output Examples: At least 5 alternative architectural plans for a sample property input, showcasing diversity in layout and design.

        *Key Features and Benefits*:
        - *Scalability*: Handles various property dimensions and constraints, making it applicable to diverse use cases.
        - *Automation*: Saves time for architects and urban planners by automating the generation and validation process.
        - *Visualization*: Provides understandable designs for clients and stakeholders.

        This project bridges the gap between regulatory compliance, generative design, and efficient visualization, offering a streamlined solution for property-based architectural planning.
        """)

    # Model section for showing plots
    elif section == "Model":
        st.header("📊 Project Data Visualization (Model)")
        st.markdown("Click the button below to display various plots related to the project.")

        if st.button("Generate Plots"):
            st.markdown("Here are 5 different plots related to the project:")

            # Generate plots
            fig1, fig2, fig3, fig4, fig5 = generate_plots()

            # Display the plots
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.pyplot(fig3)
            st.pyplot(fig4)
            st.pyplot(fig5)

# Run the app
if __name__ == "__main__":
    main()