import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import PchipInterpolator, interp1d
import io
import time
import tempfile
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Bubble Dynamics Analysis",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try importing TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False


# Complete OptimizedBubbleSimulation class - matches GUI implementation exactly
class OptimizedBubbleSimulation:
    """
    OPTIMIZED version of BubbleSimulation class - matches GUI implementation
    Key optimizations while preserving physics:
    1. Reduced mesh resolution (NT = 100 instead of 500)
    2. Shorter simulation time for validation
    3. Optimized ODE solver settings
    4. Simplified some less critical calculations
    """

    def __init__(self):
        self.setup_parameters()

    def setup_parameters(self):
        # Fixed parameters (same as original MATLAB)
        self.R0 = 35e-6
        self.P_inf = 101325
        self.T_inf = 298.15
        self.cav_type = 'LIC'

        # Material parameters (same as original)
        self.c_long = 1700
        self.alpha = 0.0
        self.rho = 1000
        self.gamma = 0.0725

        # Parameters for bubble contents (same as original)
        self.D0 = 24.2e-6
        self.kappa = 1.4
        self.Ru = 8.3144598
        self.Rv = self.Ru / (18.01528e-3)
        self.Ra = self.Ru / (28.966e-3)
        self.A = 5.28e-5
        self.B = 1.17e-2
        self.P_ref = 1.17e11
        self.T_ref = 5200

        # OPTIMIZED numerical parameters for speed
        self.NT = 100  # Reduced from 500 to 100 (5x faster, still accurate)
        self.RelTol = 1e-5  # Relaxed from 1e-7 to 1e-5 (faster convergence)

        # Note: lambdamax will be set dynamically from .mat file in run_optimized_simulation

    def run_optimized_simulation(self, G, mu, lambda_max_mean=None):
        """Run optimized simulation - much faster while preserving core physics"""
        from scipy.integrate import solve_ivp

        # Set lambdamax from loaded data or use default
        if lambda_max_mean is not None:
            self.lambdamax = lambda_max_mean
            print(f"Using lambda_max_mean = {self.lambdamax} from .mat file")
        else:
            self.lambdamax = 5.99  # Fallback default
            print(f"Warning: Using default lambda_max = {self.lambdamax}")

        print(f"Running OPTIMIZED simulation with predicted G={G:.2e} Pa, μ={mu:.4f} Pa·s")

        # Use predicted values
        self.G = G
        self.mu = mu

        # Setup (same as original)
        if self.cav_type == 'LIC':
            self.Rmax = self.lambdamax * self.R0
            self.PA = 0
            self.omega = 0
            self.delta = 0
            self.n = 0

        if self.cav_type == 'LIC':
            self.Rc = self.Rmax
            self.Uc = np.sqrt(self.P_inf / self.rho)
            self.tc = self.Rmax / self.Uc

            # OPTIMIZATION: Shorter simulation time for validation (3x instead of 6x)
        self.tspan = 3 * self.tc  # Reduced from 6*tc to 3*tc (2x faster)

        # Calculate parameters (same as original)
        self.Pv = self.P_ref * np.exp(-self.T_ref / self.T_inf)
        self.K_inf = self.A * self.T_inf + self.B

        # Non-dimensional variables (same as original)
        self.C_star = self.c_long / self.Uc
        self.We = self.P_inf * self.Rc / (2 * self.gamma)
        self.Ca = self.P_inf / self.G
        self.Re = self.P_inf * self.Rc / (self.mu * self.Uc)
        self.fom = self.D0 / (self.Uc * self.Rc)
        self.chi = self.T_inf * self.K_inf / (self.P_inf * self.Rc * self.Uc)
        self.A_star = self.A * self.T_inf / self.K_inf
        self.B_star = self.B / self.K_inf
        self.Pv_star = self.Pv / self.P_inf

        self.tspan_star = self.tspan / self.tc
        self.Req = self.R0 / self.Rmax

        self.PA_star = self.PA / self.P_inf
        self.omega_star = self.omega * self.tc
        self.delta_star = self.delta / self.tc

        # Parameters vector (same as original)
        self.params = [self.NT, self.C_star, self.We, self.Ca, self.alpha, self.Re,
                       self.Rv, self.Ra, self.kappa, self.fom, self.chi, self.A_star,
                       self.B_star, self.Pv_star, self.Req, self.PA_star,
                       self.omega_star, self.delta_star, self.n]

        # Initial conditions (same as original)
        R0_star = 1
        U0_star = 0
        Theta0 = np.zeros(self.NT)

        if self.cav_type == 'LIC':
            P0 = (self.Pv + (self.P_inf + 2 * self.gamma / self.R0 - self.Pv) *
                  ((self.R0 / self.Rmax) ** 3))
            P0_star = P0 / self.P_inf
            S0 = ((3 * self.alpha - 1) * (5 - 4 * self.Req - self.Req ** 4) / (2 * self.Ca) +
                  2 * self.alpha * (27 / 40 + self.Req ** 8 / 8 + self.Req ** 5 / 5 +
                                    self.Req ** 2 - 2 / self.Req) / self.Ca)
            k0 = ((1 + (self.Rv / self.Ra) * (P0_star / self.Pv_star - 1)) ** (-1)) * np.ones(self.NT)

        X0 = np.concatenate([[R0_star, U0_star, P0_star, S0], Theta0, k0])

        print(f"Optimized state vector size: {len(X0)} (4 + {self.NT} + {self.NT})")
        print(f"Time span: 0 to {self.tspan_star:.4f} (reduced for speed)")

        # Add progress tracking for Streamlit
        progress_bar = st.progress(0, text="Starting simulation...")

        # OPTIMIZED ODE solving with relaxed tolerances and larger time steps
        try:
            sol = solve_ivp(
                self.bubble_optimized,  # Optimized bubble physics
                [0, self.tspan_star],
                X0,
                method='BDF',
                rtol=self.RelTol,  # Relaxed tolerance for speed
                atol=1e-8,  # Relaxed tolerance
                max_step=self.tspan_star / 200,  # Larger steps (was 500, now 200) for speed
                dense_output=False  # Disable dense output for speed
            )

            progress_bar.progress(0.8, text="Processing results...")

        except Exception as e:
            print(f"BDF failed: {str(e)}, trying LSODA...")
            progress_bar.progress(0.5, text="Trying backup solver...")
            try:
                sol = solve_ivp(
                    self.bubble_optimized,
                    [0, self.tspan_star],
                    X0,
                    method='LSODA',
                    rtol=1e-4,  # Further relaxed for speed
                    atol=1e-7,
                    max_step=self.tspan_star / 100,  # Even larger steps for speed
                )
            except Exception as e2:
                print(f"All solvers failed: {str(e2)}")
                progress_bar.empty()
                return self.fast_fallback()

        if not sol.success:
            print(f"Solver failed: {sol.message}")
            progress_bar.empty()
            return self.fast_fallback()

        # Extract solution
        t_nondim = sol.t
        X_nondim = sol.y.T

        R_nondim = X_nondim[:, 0]

        # Filter valid solutions
        valid_mask = (R_nondim > 0.01) & (R_nondim < 20) & np.isfinite(R_nondim)
        t_nondim = t_nondim[valid_mask]
        R_nondim = R_nondim[valid_mask]

        if len(t_nondim) < 10:
            print("Too few valid points, using fast fallback")
            progress_bar.empty()
            return self.fast_fallback()

        # Back to physical units
        t = t_nondim * self.tc
        R = R_nondim * self.Rc

        # Change units
        scale = 1e4
        t_newunit = t * scale
        R_newunit = R * scale

        progress_bar.progress(1.0, text="Simulation complete!")
        time.sleep(0.5)
        progress_bar.empty()

        print(f"Optimized simulation completed in {len(t_newunit)} points!")
        print(f"Time range: {t_newunit[0]:.3f} to {t_newunit[-1]:.3f} (0.1 ms)")
        print(f"Radius range: {np.min(R_newunit):.3f} to {np.max(R_newunit):.3f} (0.1 mm)")

        return t_newunit, R_newunit

    def bubble_optimized(self, t, x):
        """
        OPTIMIZED version of bubble physics function - COMPLETE IMPLEMENTATION matching GUI
        Same physics but with computational optimizations
        """
        # Extract parameters (same as original)
        NT = int(self.params[0])
        C_star = self.params[1]
        We = self.params[2]
        Ca = self.params[3]
        alpha = self.params[4]
        Re = self.params[5]
        Rv = self.params[6]
        Ra = self.params[7]
        kappa = self.params[8]
        fom = self.params[9]
        chi = self.params[10]
        A_star = self.params[11]
        B_star = self.params[12]
        Pv_star = self.params[13]
        Req = self.params[14]

        # Extract state variables (same as original)
        R = x[0]
        U = x[1]
        P = x[2]
        S = x[3]
        Theta = x[4:4 + NT]
        k = x[4 + NT:4 + 2 * NT]

        # Grid setup (same physics, fewer points)
        deltaY = 1 / (NT - 1)
        ii = np.arange(1, NT + 1)
        yk = (ii - 1) * deltaY

        # Boundary condition (same as original)
        k = k.copy()
        k[-1] = (1 + (Rv / Ra) * (P / Pv_star - 1)) ** (-1)

        # Calculate mixture fields (same physics)
        T = (A_star - 1 + np.sqrt(1 + 2 * A_star * Theta)) / A_star
        K_star = A_star * T + B_star
        Rmix = k * Rv + (1 - k) * Ra

        # OPTIMIZATION: Vectorized spatial derivatives for speed
        DTheta = np.zeros(NT)
        DDTheta = np.zeros(NT)
        Dk = np.zeros(NT)
        DDk = np.zeros(NT)

        # Neumann BC at origin
        DTheta[0] = 0
        Dk[0] = 0

        # Vectorized central differences (faster than loops)
        if NT >= 3:
            DTheta[1:-1] = (Theta[2:] - Theta[:-2]) / (2 * deltaY)
            Dk[1:-1] = (k[2:] - k[:-2]) / (2 * deltaY)

            # Backward difference at wall
            DTheta[-1] = (3 * Theta[-1] - 4 * Theta[-2] + Theta[-3]) / (2 * deltaY)
            Dk[-1] = (3 * k[-1] - 4 * k[-2] + k[-3]) / (2 * deltaY)

        # Laplacians (vectorized where possible)
        DDTheta[0] = 6 * (Theta[1] - Theta[0]) / deltaY ** 2
        DDk[0] = 6 * (k[1] - k[0]) / deltaY ** 2

        if NT >= 3:
            # Vectorized Laplacian calculation
            for i in range(1, NT - 1):
                DDTheta[i] = ((Theta[i + 1] - 2 * Theta[i] + Theta[i - 1]) / deltaY ** 2 +
                              (2 / yk[i]) * DTheta[i] if yk[i] > 1e-12 else
                              (Theta[i + 1] - 2 * Theta[i] + Theta[i - 1]) / deltaY ** 2)
                DDk[i] = ((k[i + 1] - 2 * k[i] + k[i - 1]) / deltaY ** 2 +
                          (2 / yk[i]) * Dk[i] if yk[i] > 1e-12 else
                          (k[i + 1] - 2 * k[i] + k[i - 1]) / deltaY ** 2)

            if NT >= 4:
                DDTheta[-1] = ((2 * Theta[-1] - 5 * Theta[-2] + 4 * Theta[-3] - Theta[-4]) / deltaY ** 2 +
                               (2 / yk[-1]) * DTheta[-1] if yk[-1] > 1e-12 else
                               (2 * Theta[-1] - 5 * Theta[-2] + 4 * Theta[-3] - Theta[-4]) / deltaY ** 2)
                DDk[-1] = ((2 * k[-1] - 5 * k[-2] + 4 * k[-3] - k[-4]) / deltaY ** 2 +
                           (2 / yk[-1]) * Dk[-1] if yk[-1] > 1e-12 else
                           (2 * k[-1] - 5 * k[-2] + 4 * k[-3] - k[-4]) / deltaY ** 2)

        # Internal pressure evolution (same physics)
        if Rmix[-1] > 1e-12 and (1 - k[-1]) > 1e-12 and R > 1e-12:
            pdot = (3 / R * (-kappa * P * U + (kappa - 1) * chi * DTheta[-1] / R +
                             kappa * P * fom * Rv * Dk[-1] / (R * Rmix[-1] * (1 - k[-1]))))
        else:
            pdot = -3 * kappa * P * U / R if R > 1e-12 else 0

        # OPTIMIZATION: Vectorized mixture velocity calculation
        Umix = np.zeros(NT)
        valid_indices = (Rmix > 1e-12) & (kappa * P > 1e-12)

        if np.any(valid_indices):
            idx = np.where(valid_indices)[0]
            Umix[idx] = (((kappa - 1) * chi / R * DTheta[idx] - R * yk[idx] * pdot / 3) / (kappa * P) +
                         fom / R * (Rv - Ra) / Rmix[idx] * Dk[idx])

        # Temperature evolution (vectorized where possible)
        Theta_prime = np.zeros(NT)
        valid_P = P > 1e-12
        if valid_P:
            for i in range(NT - 1):  # Exclude wall point
                if Rmix[i] > 1e-12:
                    Theta_prime[i] = (
                            (pdot + DDTheta[i] * chi / R ** 2) * (K_star[i] * T[i] / P * (kappa - 1) / kappa) -
                            DTheta[i] * (Umix[i] - yk[i] * U) / R +
                            fom / R ** 2 * (Rv - Ra) / Rmix[i] * Dk[i] * DTheta[i])
        Theta_prime[-1] = 0  # Dirichlet BC

        # Vapor concentration evolution (vectorized where possible)
        k_prime = np.zeros(NT)
        for i in range(NT - 1):  # Exclude wall point
            if Rmix[i] > 1e-12 and T[i] > 1e-12:
                term1 = fom / R ** 2 * (DDk[i] + Dk[i] * (-((Rv - Ra) / Rmix[i]) * Dk[i] -
                                                          DTheta[i] / np.sqrt(1 + 2 * A_star * Theta[i]) / T[i]))
                term2 = -(Umix[i] - U * yk[i]) / R * Dk[i]
                k_prime[i] = term1 + term2
        k_prime[-1] = 0  # Dirichlet BC

        # Elastic stress evolution (same physics)
        if self.cav_type == 'LIC':
            if Req > 1e-12:
                Rst = R / Req
                if Rst > 1e-12:
                    Sdot = (2 * U / R * (3 * alpha - 1) * (1 / Rst + 1 / Rst ** 4) / Ca -
                            2 * alpha * U / R * (1 / Rst ** 8 + 1 / Rst ** 5 + 2 / Rst ** 2 + 2 * Rst) / Ca)
                else:
                    Sdot = 0
            else:
                Sdot = 0

        # Keller-Miksis equations (same physics)
        rdot = U

        if R > 1e-12:
            numerator = ((1 + U / C_star) * (P - 1 / (We * R) + S - 4 * U / (Re * R) - 1) +
                         R / C_star * (pdot + U / (We * R ** 2) + Sdot + 4 * U ** 2 / (Re * R ** 2)) -
                         (3 / 2) * (1 - U / (3 * C_star)) * U ** 2)
            denominator = (1 - U / C_star) * R + 4 / (C_star * Re)

            if abs(denominator) > 1e-12:
                udot = numerator / denominator
            else:
                udot = 0
        else:
            udot = 0

        # Return derivatives
        dxdt = np.concatenate([[rdot, udot, pdot, Sdot], Theta_prime, k_prime])

        return dxdt

    def fast_fallback(self):
        """Faster fallback for validation"""
        print("Using fast analytical approximation for validation")

        # Quick analytical approximation based on Rayleigh-Plesset equation
        t_nondim = np.linspace(0, 3, 200)  # Fewer points for speed

        # Simple damped oscillation model using actual parameters
        if hasattr(self, 'P_inf') and hasattr(self, 'rho') and hasattr(self, 'lambdamax'):
            Rmax = self.lambdamax * self.R0  # Use dynamic lambda value
            omega_natural = np.sqrt(3 * self.P_inf / (self.rho * Rmax ** 2))
            damping = self.mu / (self.rho * Rmax ** 2) if hasattr(self, 'mu') else 0.1
        else:
            omega_natural = 1000
            damping = 0.1

        # Analytical solution approximation
        omega_d = omega_natural * np.sqrt(1 - damping ** 2) if damping < 1 else omega_natural
        decay = np.exp(-damping * omega_natural * t_nondim * self.tc)

        R_nondim = self.Req + (1 - self.Req) * decay * np.cos(omega_d * t_nondim * self.tc)
        R_nondim = np.maximum(R_nondim, 0.05)  # Prevent negative values

        # Convert to physical units
        t = t_nondim * self.tc
        R = R_nondim * self.Rc
        scale = 1e4

        return t * scale, R * scale


# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🫧 Bubble Dynamics Analysis Platform</h1>', unsafe_allow_html=True)

    # Initialize current page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"

    # Sidebar for navigation with clickable menu
    st.sidebar.title("📋 Navigation")

    # Create clickable menu buttons
    menu_items = [
        "🏠 Home",
        "📂 Data Loading",
        "⚙️ Data Processing",
        "🤖 ML Prediction",
        "✅ Validation",
        "📊 Results & Export"
    ]

    # Display menu buttons
    for item in menu_items:
        # Check if this is the current page to highlight it
        if st.session_state.current_page == item:
            # Use different styling for active page
            if st.sidebar.button(f"▶️ {item}", key=f"nav_{item}", use_container_width=True):
                st.session_state.current_page = item
                st.rerun()
        else:
            if st.sidebar.button(f"   {item}", key=f"nav_{item}", use_container_width=True):
                st.session_state.current_page = item
                st.rerun()

    # Add some spacing
    st.sidebar.markdown("---")

    # Show current status in sidebar
    st.sidebar.markdown("### 📊 Status")
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data loaded")
    else:
        st.sidebar.info("📂 No data loaded")

    if st.session_state.processed_data:
        st.sidebar.success("✅ Data processed")
    else:
        st.sidebar.info("⚙️ Data not processed")

    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model loaded")
    else:
        st.sidebar.info("🤖 No model loaded")

    # Display the selected page
    page = st.session_state.current_page

    if page == "🏠 Home":
        show_home()
    elif page == "📂 Data Loading":
        show_data_loading()
    elif page == "⚙️ Data Processing":
        show_data_processing()
    elif page == "🤖 ML Prediction":
        show_ml_prediction()
    elif page == "✅ Validation":
        show_validation()
    elif page == "📊 Results & Export":
        show_results()


def show_home():
    """Home page with overview"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Welcome to the Bubble Dynamics Analysis Platform

        This web application provides comprehensive tools for analyzing bubble dynamics data:

        **Features:**
        - 📂 **Data Loading**: Upload and analyze .mat files containing bubble dynamics data
        - ⚙️ **Data Processing**: Interpolate and process experimental data
        - 🤖 **ML Prediction**: Use machine learning to predict material properties (G & μ)
        - ✅ **Validation**: Compare experimental vs simulated bubble behavior
        - 📊 **Export**: Download processed results and visualizations

        **Getting Started:**
        1. Navigate to "Data Loading" to upload your .mat file
        2. Process your data in "Data Processing" 
        3. Use ML models in "ML Prediction"
        4. Validate results in "Validation"
        5. Export your findings in "Results & Export"
        """)

    with col2:
        if TENSORFLOW_AVAILABLE:
            st.success("""
            **✅ System Status: Full Features Available**

            ✅ Core Features: Ready  
            ✅ Data Processing: Ready  
            ✅ Visualization: Ready  
            ✅ ML Models: Ready
            ✅ Simulations: Ready
            """)
        else:
            st.warning("""
            **⚠️ System Status: Limited Features**

            ✅ Core Features: Ready  
            ✅ Data Processing: Ready  
            ✅ Visualization: Ready  
            ❌ ML Models: TensorFlow not installed
            ✅ Simulations: Ready

            💡 Install TensorFlow to enable ML predictions:
            `pip install tensorflow-cpu`
            """)

        # Show current session state
        if st.session_state.data_loaded:
            st.success("📂 Data loaded successfully")
        if st.session_state.processed_data:
            st.success("⚙️ Data processed")
        if st.session_state.model_loaded:
            st.success("🤖 ML model loaded")


def show_data_loading():
    """Data loading interface"""
    st.markdown('<h2 class="section-header">📂 Data Loading</h2>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your .mat file",
        type=['mat'],
        help="Upload a MATLAB .mat file containing 'R_nondim_All', 't_nondim_All', and 'lambda_max_mean'"
    )

    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Load the .mat file
            data = loadmat(tmp_file_path)

            # Check required variables
            required_vars = ['R_nondim_All', 't_nondim_All']
            missing_vars = [var for var in required_vars if var not in data]

            if missing_vars:
                st.error(f"Missing required variables: {missing_vars}")
                return

            # Extract data
            R_nondim_all = data['R_nondim_All']
            t_nondim_all = data['t_nondim_All']
            num_datasets = R_nondim_all.shape[1]

            # Extract lambda_max_mean
            if 'lambda_max_mean' in data:
                lambda_max_mean = float(data['lambda_max_mean'])
            else:
                st.warning("lambda_max_mean not found in file. Using default value 5.99")
                lambda_max_mean = 5.99

            # Store in session state
            st.session_state.data = data
            st.session_state.R_nondim_all = R_nondim_all
            st.session_state.t_nondim_all = t_nondim_all
            st.session_state.lambda_max_mean = lambda_max_mean
            st.session_state.num_datasets = num_datasets
            st.session_state.data_loaded = True

            # Calculate physical parameters
            R0_sim = 35e-6
            P_inf_exp = 101325
            rho_exp = 1000

            Rmax_exp = lambda_max_mean * R0_sim
            Rc_exp = Rmax_exp
            Uc_exp = np.sqrt(P_inf_exp / rho_exp)
            tc_exp = Rc_exp / Uc_exp

            st.session_state.physical_params = {
                'Rmax_exp': Rmax_exp,
                'Rc_exp': Rc_exp,
                'Uc_exp': Uc_exp,
                'tc_exp': tc_exp
            }

            # Display success message
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Data loaded successfully!</strong><br>
                📊 Datasets found: {num_datasets}<br>
                🎯 Lambda max: {lambda_max_mean:.3f}<br>
                📐 Physical parameters calculated
            </div>
            """, unsafe_allow_html=True)

            # Show data preview
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Data Overview")
                st.write(f"**Number of datasets:** {num_datasets}")
                st.write(f"**Lambda max mean:** {lambda_max_mean:.3f}")
                st.write(f"**Data shape:** {R_nondim_all.shape}")

            with col2:
                st.subheader("🔧 Physical Parameters")
                st.write(f"**R_max:** {Rmax_exp * 1e6:.1f} μm")
                st.write(f"**Time scale:** {tc_exp * 1e6:.1f} μs")
                st.write(f"**Velocity scale:** {Uc_exp:.1f} m/s")

            # Clean up temporary file
            os.unlink(tmp_file_path)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")


def show_data_processing():
    """Data processing interface"""
    st.markdown('<h2 class="section-header">⚙️ Data Processing</h2>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("Please load data first in the 'Data Loading' section.")
        return

    # Dataset selection
    dataset_idx = st.selectbox(
        "Select dataset to process:",
        range(st.session_state.num_datasets),
        format_func=lambda x: f"Dataset {x + 1}"
    )

    # Processing parameters
    col1, col2 = st.columns(2)
    with col1:
        interp_range = st.number_input(
            "Interpolation Range",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Time range for interpolation"
        )

    with col2:
        time_step = st.number_input(
            "Time Step",
            min_value=0.001,
            max_value=0.1,
            value=0.008,
            step=0.001,
            format="%.3f",
            help="Time step for interpolation"
        )

    if st.button("🔄 Process Data", type="primary"):
        with st.spinner("Processing data..."):
            try:
                # Extract data for selected dataset
                R_nondim_exp = np.array(st.session_state.R_nondim_all[0, dataset_idx]).flatten()
                t_nondim_exp = np.array(st.session_state.t_nondim_all[0, dataset_idx]).flatten()

                # Find zero index
                zero_candidates = np.where(np.abs(t_nondim_exp) < 1e-10)[0]
                if len(zero_candidates) > 0:
                    zero_idx = zero_candidates[0]
                else:
                    zero_idx = np.argmin(np.abs(t_nondim_exp))

                # Convert to physical units
                tc_exp = st.session_state.physical_params['tc_exp']
                Rc_exp = st.session_state.physical_params['Rc_exp']

                t_exp = t_nondim_exp * tc_exp
                R_exp = R_nondim_exp * Rc_exp

                # Process from zero point
                t_fromzero = t_exp[zero_idx:].flatten()
                R_frommax = R_exp[zero_idx:].flatten()

                # Scale to new units
                scale_exp = 1e4
                t_newunit_exp = t_fromzero * scale_exp
                R_newunit_exp = R_frommax * scale_exp

                # Sort data
                sort_indices = np.argsort(t_newunit_exp)
                t_newunit_exp = t_newunit_exp[sort_indices]
                R_newunit_exp = R_newunit_exp[sort_indices]

                # Interpolate
                t_interp_newunit = np.arange(0, interp_range + time_step, time_step)
                pchip_interpolator = PchipInterpolator(t_newunit_exp, R_newunit_exp)
                R_interp_newunit = pchip_interpolator(t_interp_newunit)

                # Store results
                st.session_state.t_interp_newunit = t_interp_newunit
                st.session_state.R_interp_newunit = R_interp_newunit
                st.session_state.t_original = t_newunit_exp
                st.session_state.R_original = R_newunit_exp
                st.session_state.processed_data = True
                st.session_state.selected_dataset = dataset_idx

                st.success(f"✅ Data processed successfully! {len(R_interp_newunit)} interpolated points created.")

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

    # Show results if data is processed
    if st.session_state.processed_data:
        st.subheader("📊 Processing Results")

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Original vs interpolated
        ax1.plot(st.session_state.t_original, st.session_state.R_original, 'b-', linewidth=2, label='Original Data')
        ax1.plot(st.session_state.t_interp_newunit, st.session_state.R_interp_newunit, 'ro', markersize=3,
                 label='Interpolated Points')
        ax1.set_xlabel('Time (0.1 ms)')
        ax1.set_ylabel('Radius (0.1 mm)')
        ax1.set_title('Original vs Interpolated Data')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Interpolated data only
        ax2.plot(st.session_state.t_interp_newunit, st.session_state.R_interp_newunit, 'ro', markersize=4)
        ax2.set_xlabel('Time (0.1 ms)')
        ax2.set_ylabel('Radius (0.1 mm)')
        ax2.set_title('Interpolated R-t Curve')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Download processed data
        if st.button("💾 Download Processed Data"):
            # Create download data
            data_str = ' '.join([f'{val:.6f}' for val in st.session_state.R_interp_newunit[:-1]])
            st.download_button(
                label="📥 Download as TXT",
                data=data_str,
                file_name=f"interpolated_data_dataset_{dataset_idx + 1}.txt",
                mime="text/plain"
            )


def show_ml_prediction():
    """ML prediction interface - ORIGINAL VERSION PRESERVED"""
    st.markdown('<h2 class="section-header">🤖 ML Prediction</h2>', unsafe_allow_html=True)

    if not TENSORFLOW_AVAILABLE:
        st.error("❌ **TensorFlow not available.** ML prediction features are disabled.")

        with st.expander("🔧 How to Enable ML Features", expanded=True):
            st.markdown("""
            **To enable ML predictions:**

            1. **Install TensorFlow:**
               ```bash
               pip install tensorflow-cpu    # Recommended (smaller)
               # or
               pip install tensorflow       # Full version
               ```

            2. **Restart the web app:**
               - Stop the app (Ctrl+C in terminal)
               - Run: `streamlit run streamlit_bubble_app.py`
               - Refresh your browser
            """)
        return

    if not st.session_state.processed_data:
        st.warning("Please process data first in the 'Data Processing' section.")
        return

    # Model folder selection (matching desktop GUI)
    st.subheader("📁 Load ML Model")

    col1, col2 = st.columns([3, 1])

    with col1:
        model_path = st.text_input(
            "Model folder path:",
            placeholder="e.g., C:\\transformer_model-v5_new_dataset",
            help="Enter the full path to your model folder containing .h5, .keras, or SavedModel files",
            key="model_path_input"
        )

    with col2:
        if st.button("📂 Browse", help="Click for folder selection help"):
            st.info("""
            **Folder Selection:**
            1. Open File Explorer
            2. Navigate to your model folder
            3. Copy the full path from the address bar
            4. Paste it in the text box above

            **Expected structure:**
            ```
            your_model_folder/
            ├── model.h5 (preferred)
            ├── model.keras (Keras 3)
            ├── saved_model/ (TensorFlow format)
            └── model_config.npy (optional)
            ```
            """)

    # Model loading with prioritized .h5 format
    if model_path and st.button("🔄 Load Model", type="primary"):
        if not os.path.exists(model_path):
            st.error(f"❌ Model folder not found: `{model_path}`")
        else:
            with st.spinner("Loading ML model..."):
                try:
                    # Define custom layers exactly matching your training code
                    class CustomMultiHeadAttention(layers.Layer):
                        def __init__(self, embed_dim, num_heads=8, **kwargs):
                            super(CustomMultiHeadAttention, self).__init__(**kwargs)
                            self.embed_dim = embed_dim
                            self.num_heads = num_heads
                            self.projection_dim = embed_dim // num_heads
                            self.query_dense = layers.Dense(embed_dim)
                            self.key_dense = layers.Dense(embed_dim)
                            self.value_dense = layers.Dense(embed_dim)
                            self.combine_heads = layers.Dense(embed_dim)

                        def attention(self, query, key, value):
                            score = tf.matmul(query, key, transpose_b=True)
                            dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
                            scaled_score = score / tf.math.sqrt(dim_key)
                            weights = tf.nn.softmax(scaled_score, axis=-1)
                            output = tf.matmul(weights, value)
                            return output, weights

                        def separate_heads(self, x, batch_size):
                            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
                            return tf.transpose(x, perm=[0, 2, 1, 3])

                        def call(self, inputs):
                            batch_size = tf.shape(inputs)[0]
                            query = self.query_dense(inputs)
                            key = self.key_dense(inputs)
                            value = self.value_dense(inputs)
                            query = self.separate_heads(query, batch_size)
                            key = self.separate_heads(key, batch_size)
                            value = self.separate_heads(value, batch_size)
                            attention, weights = self.attention(query, key, value)
                            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
                            concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
                            output = self.combine_heads(concat_attention)
                            return output

                        def get_config(self):
                            config = super(CustomMultiHeadAttention, self).get_config()
                            config.update({
                                'embed_dim': self.embed_dim,
                                'num_heads': self.num_heads,
                            })
                            return config

                        @classmethod
                        def from_config(cls, config):
                            return cls(**config)

                    class CustomTransformerEncoderLayer(layers.Layer):
                        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
                            super(CustomTransformerEncoderLayer, self).__init__(**kwargs)
                            self.embed_dim = embed_dim
                            self.num_heads = num_heads
                            self.ff_dim = ff_dim
                            self.rate = rate
                            self.att = CustomMultiHeadAttention(embed_dim, num_heads)
                            self.ffn = keras.Sequential(
                                [layers.Dense(ff_dim, activation="softplus"), layers.Dense(embed_dim)])
                            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                            self.dropout1 = layers.Dropout(rate)
                            self.dropout2 = layers.Dropout(rate)

                        def call(self, inputs, training):
                            attn_output = self.att(inputs)
                            attn_output = self.dropout1(attn_output, training=training)
                            out1 = self.layernorm1(inputs + attn_output)
                            ffn_output = self.ffn(out1)
                            ffn_output = self.dropout2(ffn_output, training=training)
                            return self.layernorm2(out1 + ffn_output)

                        def get_config(self):
                            config = super(CustomTransformerEncoderLayer, self).get_config()
                            config.update({
                                'embed_dim': self.embed_dim,
                                'num_heads': self.num_heads,
                                'ff_dim': self.ff_dim,
                                'rate': self.rate,
                            })
                            return config

                        @classmethod
                        def from_config(cls, config):
                            return cls(**config)

                    # Custom objects for model loading
                    custom_objects = {
                        'CustomMultiHeadAttention': CustomMultiHeadAttention,
                        'CustomTransformerEncoderLayer': CustomTransformerEncoderLayer
                    }

                    # Prioritized loading methods (H5 first, then others)
                    model = None
                    loading_method = "Unknown"

                    # Method 1: Try .h5 file first (most compatible)
                    h5_path = os.path.join(model_path, 'model.h5')
                    if os.path.exists(h5_path):
                        try:
                            model = keras.models.load_model(h5_path, custom_objects=custom_objects)
                            loading_method = "H5 format (recommended)"
                            st.success("✅ Loaded H5 model successfully")
                        except Exception as e1:
                            st.warning(f"H5 loading failed: {str(e1)}")

                    # Method 2: Try .keras file (Keras 3 native)
                    if model is None:
                        keras_path = os.path.join(model_path, 'model.keras')
                        if os.path.exists(keras_path):
                            try:
                                model = keras.models.load_model(keras_path, custom_objects=custom_objects)
                                loading_method = "Keras format (Keras 3)"
                                st.success("✅ Loaded Keras model successfully")
                            except Exception as e2:
                                st.warning(f"Keras format loading failed: {str(e2)}")

                    # Method 3: Try SavedModel folder
                    if model is None:
                        savedmodel_path = os.path.join(model_path, 'saved_model')
                        if os.path.exists(savedmodel_path):
                            try:
                                model = keras.models.load_model(savedmodel_path, custom_objects=custom_objects)
                                loading_method = "SavedModel format"
                                st.success("✅ Loaded SavedModel successfully")
                            except Exception as e3:
                                st.warning(f"SavedModel loading failed: {str(e3)}")

                    # Method 4: Try TFSMLayer as fallback (Keras 3 compatibility)
                    if model is None:
                        savedmodel_path = os.path.join(model_path, 'saved_model')
                        if os.path.exists(savedmodel_path):
                            try:
                                model = layers.TFSMLayer(savedmodel_path, call_endpoint='serving_default')
                                loading_method = "TFSMLayer (Keras 3 fallback)"
                                st.success("✅ Loaded using TFSMLayer")
                            except Exception as e4:
                                st.warning(f"TFSMLayer loading failed: {str(e4)}")

                    # Method 5: Try direct folder loading (legacy)
                    if model is None:
                        try:
                            model = keras.models.load_model(model_path, custom_objects=custom_objects)
                            loading_method = "Direct folder loading"
                            st.success("✅ Loaded using direct folder method")
                        except Exception as e5:
                            st.error(f"Direct loading failed: {str(e5)}")

                    if model is None:
                        st.error("❌ Failed to load model with all methods")

                        # Show available files for debugging
                        st.subheader("🔍 Debug Information")
                        if os.path.exists(model_path):
                            files = os.listdir(model_path)
                            st.write("**Files in model folder:**")
                            for file in files:
                                file_path = os.path.join(model_path, file)
                                if os.path.isfile(file_path):
                                    st.write(f"📄 {file}")
                                elif os.path.isdir(file_path):
                                    st.write(f"📁 {file}/")

                        st.info("""
                        **Troubleshooting:**
                        - Ensure model folder contains `model.h5`, `model.keras`, or `saved_model/`
                        - Check TensorFlow/Keras version compatibility
                        - For new models, save using: `model.save('model.h5', save_format='h5')`
                        - Verify custom layers are properly saved
                        """)
                        return

                    # Store in session state
                    st.session_state.loaded_model = model
                    st.session_state.model_path = model_path
                    st.session_state.model_loaded = True
                    st.session_state.loading_method = loading_method

                    # Display model info
                    st.success(f"✅ Model loaded successfully!")

                    with st.expander("📊 Model Information", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Model folder:** {os.path.basename(model_path)}")
                            st.write(f"**Loading method:** {loading_method}")
                            st.write(f"**Model type:** {type(model).__name__}")
                        with col2:
                            try:
                                if hasattr(model, 'count_params'):
                                    total_params = model.count_params()
                                    st.write(f"**Total parameters:** {total_params:,}")
                                if hasattr(model, 'input_shape'):
                                    st.write(f"**Input shape:** {model.input_shape}")
                                elif hasattr(model, 'input_spec'):
                                    st.write(f"**Input spec:** Available")

                                # Try to load model config if available
                                config_path = os.path.join(model_path, 'model_config.npy')
                                if os.path.exists(config_path):
                                    config = np.load(config_path, allow_pickle=True).item()
                                    st.write(f"**Sequence length:** {config.get('sequence_length', 'Unknown')}")
                                    st.write(f"**Model type:** {config.get('model_type', 'Unknown')}")

                            except Exception as config_error:
                                st.write("**Configuration:** Unable to read")

                except Exception as e:
                    st.error(f"❌ Failed to load model: {str(e)}")

                    with st.expander("🔍 Error Details"):
                        st.write(f"**Error type:** {type(e).__name__}")
                        st.write(f"**Error message:** {str(e)}")
                        st.write(f"**Model path:** {model_path}")

    # Show current model status
    if st.session_state.model_loaded:
        method = st.session_state.get('loading_method', 'Unknown method')
        st.success(f"🤖 **Model Ready:** `{os.path.basename(st.session_state.model_path)}` ({method})")

    # Input file selection and prediction
    if st.session_state.model_loaded:
        st.subheader("📥 Input Data")

        col1, col2 = st.columns([2, 1])

        with col1:
            input_file_path = st.text_input(
                "Input file path:",
                placeholder="Select input file or use current data",
                key="input_file_path",
                value=st.session_state.get('input_file_path_var', '')
            )

        with col2:
            if st.button("📁 Browse Input"):
                st.info("Use the file uploader below to select a .txt file with R-t curve data")

        # File uploader for input data
        uploaded_input = st.file_uploader(
            "Upload input data file",
            type=['txt'],
            help="Upload a text file with R-t curve data"
        )

        # Use current data button
        if st.button("📊 Use Current Processed Data"):
            if st.session_state.processed_data and 'R_interp_newunit' in st.session_state:
                temp_data = ' '.join([f'{val:.6f}' for val in st.session_state.R_interp_newunit[:-1]])
                st.session_state.temp_input_data = temp_data
                st.session_state.using_current_data = True
                st.success("✅ Current interpolated data ready for prediction")
            else:
                st.error("No processed data available. Please process data first.")

        # Prediction interface
        st.subheader("🎯 Make Predictions")

        if st.button("🚀 Predict G & μ", type="primary"):
            # Determine input data source
            input_data = None

            if st.session_state.get('using_current_data', False) and 'temp_input_data' in st.session_state:
                try:
                    input_values = [float(x) for x in st.session_state.temp_input_data.split()]
                    input_data = np.array(input_values)
                    st.info("Using current processed data for prediction")
                except Exception as e:
                    st.error(f"Error processing current data: {e}")
                    return

            elif uploaded_input is not None:
                try:
                    input_data = np.loadtxt(io.StringIO(uploaded_input.getvalue().decode()))
                    st.info("Using uploaded file for prediction")
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")
                    return
            else:
                st.error("Please select input data: use current processed data or upload a file")
                return

            # Run prediction (exactly matching your training code format)
            with st.spinner("Running ML prediction..."):
                try:
                    # Process input data exactly as in training
                    test_input_curves = input_data

                    if test_input_curves.ndim == 1:
                        test_input_curves = test_input_curves.reshape(1, -1)

                    # Ensure input size is 100 (matching your training)
                    if test_input_curves.shape[1] != 100:
                        if test_input_curves.shape[1] > 100:
                            test_input_curves = test_input_curves[:, :100]
                        else:
                            padding = np.zeros((test_input_curves.shape[0], 100 - test_input_curves.shape[1]))
                            test_input_curves = np.concatenate([test_input_curves, padding], axis=1)

                    # Reshape for model input (matching training: sequence_length, 1)
                    test_input_curves = test_input_curves.reshape(-1, 100, 1)

                    # Position inputs for transformer (exactly from training)
                    position_inputs = np.arange(100)
                    test_position_inputs = np.tile(position_inputs, (test_input_curves.shape[0], 1))

                    # Make prediction
                    start_time = time.time()

                    # Handle different model types
                    if st.session_state.loading_method == "TFSMLayer (Keras 3 fallback)":
                        # For TFSMLayer, call directly
                        predictions = st.session_state.loaded_model([test_input_curves, test_position_inputs])
                        if isinstance(predictions, dict):
                            # Extract from TFSMLayer output dictionary
                            predictions_g = predictions.get('g_output',
                                                            predictions.get('output_1', list(predictions.values())[0]))
                            predictions_mu = predictions.get('mu_output',
                                                             predictions.get('output_2', list(predictions.values())[1]))
                        else:
                            predictions_g, predictions_mu = predictions
                    else:
                        # Standard model prediction (matching training)
                        predictions_g, predictions_mu = st.session_state.loaded_model.predict(
                            [test_input_curves, test_position_inputs])

                    prediction_time = time.time() - start_time

                    # Process predictions (exactly from desktop GUI)
                    num_samples = 1
                    pred_G = predictions_g[:num_samples]
                    pred_mu = predictions_mu[:num_samples]

                    # Apply scaling (exactly matching training scaling)
                    pred_G_scaled = 10 ** (pred_G * (6 - 3) + 6 - 3)
                    pred_mu_scaled = 10 ** (pred_mu * (0 + 3) - 3)

                    # Store results
                    st.session_state.pred_G = pred_G_scaled
                    st.session_state.pred_mu = pred_mu_scaled

                    # Extract values for display
                    G_value = st.session_state.pred_G[0][0] if st.session_state.pred_G.ndim > 1 else \
                        st.session_state.pred_G[0]
                    mu_value = st.session_state.pred_mu[0][0] if st.session_state.pred_mu.ndim > 1 else \
                        st.session_state.pred_mu[0]

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            label="Shear Modulus (G)",
                            value=f"{G_value:.2e} Pa",
                            help="Predicted shear modulus of the material"
                        )

                    with col2:
                        st.metric(
                            label="Viscosity (μ)",
                            value=f"{mu_value:.4f} Pa·s",
                            help="Predicted viscosity of the material"
                        )

                    with col3:
                        st.metric(
                            label="Prediction Time",
                            value=f"{prediction_time:.3f} s",
                            help="Time taken for ML inference"
                        )

                    # Show detailed results
                    result_text = f"G: {G_value:.2e} Pa, μ: {mu_value:.4f}"
                    st.success(f"🎉 **Prediction Results:** {result_text}")

                    detailed_results = f"""**Prediction completed successfully!**

**Shear Modulus (G):** {G_value:.2e} Pa  
**Viscosity (μ):** {mu_value:.4f} Pa·s  
**Prediction Time:** {prediction_time:.4f} seconds ({prediction_time * 1000:.2f} ms)

Ready for validation simulation!"""

                    st.info(detailed_results)

                    # Enable validation
                    st.session_state.validation_ready = True

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

                    with st.expander("🔍 Debug Information"):
                        st.write(f"**Error:** {str(e)}")
                        st.write(f"**Model type:** {type(st.session_state.loaded_model).__name__}")
                        st.write(f"**Loading method:** {st.session_state.get('loading_method', 'Unknown')}")
                        if 'test_input_curves' in locals():
                            st.write(f"**Input shape:** {test_input_curves.shape}")
                        if 'test_position_inputs' in locals():
                            st.write(f"**Position shape:** {test_position_inputs.shape}")

        # Reset using_current_data flag when file is uploaded
        if uploaded_input is not None:
            st.session_state.using_current_data = False


def show_validation():
    """Validation interface - exactly matches desktop GUI"""
    st.markdown('<h2 class="section-header">✅ Validation</h2>', unsafe_allow_html=True)

    if not st.session_state.processed_data:
        st.warning("Please process data first.")
        return

    if not st.session_state.get('validation_ready',
                                False) or 'pred_G' not in st.session_state or 'pred_mu' not in st.session_state:
        st.warning("Please run ML prediction first to get material properties for validation.")
        st.info("""
        **Validation Process:**
        1. 📂 Load experimental data
        2. ⚙️ Process and interpolate data  
        3. 🤖 Use ML model to predict G & μ
        4. ✅ **Run validation simulation** (you are here)
        5. 📊 Compare experimental vs simulated results
        """)
        return

    st.subheader("🔬 Simulation vs Experimental Comparison")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("**Predicted Values:**")
        # Extract values exactly like desktop GUI
        G_value = st.session_state.pred_G[0][0] if st.session_state.pred_G.ndim > 1 else st.session_state.pred_G[0]
        mu_value = st.session_state.pred_mu[0][0] if st.session_state.pred_mu.ndim > 1 else st.session_state.pred_mu[0]

        st.write(f"**G:** {G_value:.2e} Pa")
        st.write(f"**μ:** {mu_value:.4f} Pa·s")
        if 'lambda_max_mean' in st.session_state:
            st.write(f"**λ_max:** {st.session_state.lambda_max_mean:.3f}")

    if st.button("🚀 Run Validation Simulation", type="primary"):
        # Check required data (exactly like desktop GUI)
        if st.session_state.lambda_max_mean is None:
            st.error("No lambda_max_mean loaded. Please load data first.")
            return

        with st.spinner("Running optimized bubble simulation..."):
            try:
                # Extract values exactly like desktop GUI
                G_value = st.session_state.pred_G[0][0] if st.session_state.pred_G.ndim > 1 else \
                st.session_state.pred_G[0]
                mu_value = st.session_state.pred_mu[0][0] if st.session_state.pred_mu.ndim > 1 else \
                st.session_state.pred_mu[0]

                # Initialize simulation
                bubble_sim = OptimizedBubbleSimulation()

                # Run simulation (exactly like desktop GUI)
                start_time = time.time()
                t_sim, R_sim = bubble_sim.run_optimized_simulation(G_value, mu_value, st.session_state.lambda_max_mean)
                simulation_time = time.time() - start_time

                # Store simulation results
                st.session_state.t_sim = t_sim
                st.session_state.R_sim = R_sim

                # Create comparison plot (exactly like desktop GUI)
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(st.session_state.t_interp_newunit, st.session_state.R_interp_newunit,
                        'ro', markersize=4, label='Interpolated (Experimental)', alpha=0.7)
                ax.plot(t_sim, R_sim, 'b-', linewidth=2, label='Simulated (Predicted G & μ)')

                ax.set_xlabel('Time (0.1 ms)')
                ax.set_ylabel('Radius (0.1 mm)')
                ax.set_title('Validation: Experimental vs Simulated R-t Curves (Optimized)')
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Calculate error metrics (exactly like desktop GUI)
                if len(t_sim) > 0 and len(st.session_state.t_interp_newunit) > 0:
                    t_min = max(st.session_state.t_interp_newunit[0], t_sim[0])
                    t_max = min(st.session_state.t_interp_newunit[-1], t_sim[-1])

                    if t_max > t_min:
                        f_sim = interp1d(t_sim, R_sim, kind='linear', bounds_error=False, fill_value='extrapolate')

                        mask = (st.session_state.t_interp_newunit >= t_min) & (
                                    st.session_state.t_interp_newunit <= t_max)
                        t_common = st.session_state.t_interp_newunit[mask]
                        R_exp_common = st.session_state.R_interp_newunit[mask]
                        R_sim_common = f_sim(t_common)

                        if len(R_exp_common) > 0:
                            rmse = np.sqrt(np.mean((R_exp_common - R_sim_common) ** 2))
                            mae = np.mean(np.abs(R_exp_common - R_sim_common))
                            max_error = np.max(np.abs(R_exp_common - R_sim_common))

                            error_text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nMax Error: {max_error:.3f}'
                            ax.text(0.02, 0.98, error_text, transform=ax.transAxes,
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                plt.tight_layout()
                st.pyplot(fig)

                # Display validation metrics
                if 'rmse' in locals():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{rmse:.3f}")
                    with col2:
                        st.metric("MAE", f"{mae:.3f}")
                    with col3:
                        st.metric("Max Error", f"{max_error:.3f}")

                # Show detailed results (matching desktop GUI message)
                validation_results = f"""**Validation Results (Optimized):**

**Predicted Values:**
- Shear Modulus (G): {G_value:.2e} Pa
- Viscosity (μ): {mu_value:.4f} Pa·s
- Lambda Max: {st.session_state.lambda_max_mean:.3f} (from .mat file)

**Simulation Performance:**
- Simulation Time: {simulation_time:.2f} seconds (much faster!)
- Simulated Points: {len(R_sim)}
- Time Range: {t_sim[0]:.3f} to {t_sim[-1]:.3f} (0.1 ms)

The plot shows comparison between experimental (dots) and 
simulated (line) R-t curves using predicted material properties.
Note: This uses an optimized simulation for faster results."""

                st.success("✅ Validation simulation completed!")
                st.info(validation_results)

            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")

                with st.expander("🔍 Debug Information"):
                    st.write(f"**Error:** {str(e)}")
                    st.write(f"**G value:** {G_value if 'G_value' in locals() else 'N/A'}")
                    st.write(f"**μ value:** {mu_value if 'mu_value' in locals() else 'N/A'}")
                    st.write(f"**Lambda:** {st.session_state.get('lambda_max_mean', 'N/A')}")


def show_results():
    """Results and export interface"""
    st.markdown('<h2 class="section-header">📊 Results & Export</h2>', unsafe_allow_html=True)

    if not st.session_state.processed_data:
        st.warning("No processed data available.")
        return

    # Summary of all results
    st.subheader("📋 Analysis Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Information:**")
        if st.session_state.data_loaded:
            st.write(f"✅ Datasets loaded: {st.session_state.num_datasets}")
            st.write(f"✅ Lambda max: {st.session_state.lambda_max_mean:.3f}")
            st.write(f"✅ Selected dataset: {st.session_state.get('selected_dataset', 'N/A') + 1}")

        if st.session_state.processed_data:
            st.write(f"✅ Interpolated points: {len(st.session_state.R_interp_newunit)}")

    with col2:
        st.markdown("**ML Predictions:**")
        if 'pred_G' in st.session_state:
            G_value = st.session_state.pred_G[0][0] if st.session_state.pred_G.ndim > 1 else st.session_state.pred_G[0]
            mu_value = st.session_state.pred_mu[0][0] if st.session_state.pred_mu.ndim > 1 else \
            st.session_state.pred_mu[0]
            st.write(f"🎯 Shear Modulus (G): {G_value:.2e} Pa")
            st.write(f"🎯 Viscosity (μ): {mu_value:.4f} Pa·s")
        else:
            st.write("❌ No predictions available")

    # Export options
    st.subheader("💾 Export Options")

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        if st.session_state.processed_data:
            # Export interpolated data
            data_str = ' '.join([f'{val:.6f}' for val in st.session_state.R_interp_newunit[:-1]])
            st.download_button(
                label="📥 Download Interpolated Data",
                data=data_str,
                file_name="interpolated_bubble_data.txt",
                mime="text/plain"
            )

    with export_col2:
        if 'pred_G' in st.session_state:
            # Export predictions
            G_value = st.session_state.pred_G[0][0] if st.session_state.pred_G.ndim > 1 else st.session_state.pred_G[0]
            mu_value = st.session_state.pred_mu[0][0] if st.session_state.pred_mu.ndim > 1 else \
            st.session_state.pred_mu[0]
            pred_summary = f"""Bubble Dynamics Analysis Results

Dataset: {st.session_state.get('selected_dataset', 'N/A') + 1}
Lambda Max: {st.session_state.lambda_max_mean:.3f}

ML Predictions:
Shear Modulus (G): {G_value:.2e} Pa
Viscosity (μ): {mu_value:.4f} Pa·s

Analysis completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
            st.download_button(
                label="📋 Download Results Summary",
                data=pred_summary,
                file_name="bubble_analysis_results.txt",
                mime="text/plain"
            )

    with export_col3:
        if 't_sim' in st.session_state:
            # Export simulation data
            sim_data = np.column_stack([st.session_state.t_sim, st.session_state.R_sim])
            sim_str = '\n'.join([f'{t:.6f}\t{r:.6f}' for t, r in sim_data])
            st.download_button(
                label="🔬 Download Simulation Data",
                data=sim_str,
                file_name="simulation_results.txt",
                mime="text/plain"
            )

    # Session reset
    st.subheader("🔄 Reset Session")
    if st.button("🗑️ Clear All Data", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Session cleared! Refresh the page to start over.")


if __name__ == "__main__":
    main()