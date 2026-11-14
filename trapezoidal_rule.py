"""
Trapezoidal Rule Explorer
An interactive Streamlit app that helps students see how the trapezoidal rule
approximates a definite integral, and how the approximation improves as n grows.

Run locally:   streamlit run trapezoid_app.py
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Trapezoidal Rule Explorer", page_icon="📐", layout="wide")

CURVE_COLOR = "#2a78d6"  # blue
TRAP_COLOR = "#eb6834"   # orange

x = sp.symbols("x")

PRESETS = {
    "x²": ("x^2", 0.0, 2.0),
    "sin(x)": ("sin(x)", 0.0, float(np.pi)),
    "eˣ": ("exp(x)", 0.0, 1.0),
    "√x": ("sqrt(x)", 0.0, 4.0),
    "1/x": ("1/x", 1.0, 3.0),
    "ln(x)": ("ln(x)", 1.0, 4.0),
    "x³ − 3x + 2": ("x^3 - 3*x + 2", -1.0, 2.0),
    "Custom…": (None, 0.0, 1.0),
}

ALLOWED_WORDS = {"x", "sin", "cos", "tan", "exp", "ln", "log", "sqrt", "pi", "e", "abs"}
SAFE_LOCALS = {
    "x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
    "ln": sp.log, "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "e": sp.E, "abs": sp.Abs,
}
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse_function(text: str):
    """Parse a student-typed expression in x, allowing only whitelisted names."""
    if not re.fullmatch(r"[0-9a-zA-Z\s\.\+\-\*/\^\(\)]+", text or ""):
        raise ValueError("Use only numbers, x, + − * / ^, parentheses and functions like sin, exp, ln, sqrt.")
    for word in re.findall(r"[a-zA-Z]+", text):
        if word not in ALLOWED_WORDS:
            raise ValueError(f"'{word}' isn't recognised. Allowed: {', '.join(sorted(ALLOWED_WORDS))}.")
    expr = parse_expr(text, local_dict=SAFE_LOCALS, global_dict={"Integer": sp.Integer, "Float": sp.Float,
                                                                  "Symbol": sp.Symbol, "Rational": sp.Rational},
                      transformations=TRANSFORMS)
    if expr.free_symbols - {x}:
        raise ValueError("The function can only use the variable x.")
    return expr


def safe_eval(f, xs):
    with np.errstate(all="ignore"):
        ys = np.asarray(f(xs), dtype=float) * np.ones_like(xs, dtype=float)
    return ys


def trapezoid(ys, h):
    return h / 2 * (ys[0] + ys[-1] + 2 * ys[1:-1].sum())


def fmt(v, d=6):
    return f"{v:.{d}f}".rstrip("0").rstrip(".") if abs(v) < 1e12 else f"{v:.4e}"


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Set up the integral")

choice = st.sidebar.selectbox("Function f(x)", list(PRESETS.keys()))
preset_text, default_a, default_b = PRESETS[choice]

if preset_text is None:
    func_text = st.sidebar.text_input("Type f(x)", value="x^2 + 1",
                                      help="Examples: x^3 - 2x, sin(x) + 1, exp(-x^2), sqrt(x+1)")
else:
    func_text = preset_text

# Reset a and b when the function changes
if st.session_state.get("last_choice") != choice:
    st.session_state["a"] = default_a
    st.session_state["b"] = default_b
    st.session_state["last_choice"] = choice

col_a, col_b = st.sidebar.columns(2)
a = col_a.number_input("Lower limit a", key="a", step=0.5, format="%.3f")
b = col_b.number_input("Upper limit b", key="b", step=0.5, format="%.3f")

n = st.sidebar.slider("Number of trapezoids n", min_value=1, max_value=20, value=4)

st.sidebar.markdown("---")
show_area = st.sidebar.checkbox("Shade the true area under the curve", value=False)
show_points = st.sidebar.checkbox("Show the points (xᵢ, yᵢ)", value=True)

# ---------------------------------------------------------------------------
# Validate input
# ---------------------------------------------------------------------------
st.title("📐 The Trapezoidal Rule")
st.markdown(
    "Approximate the area under a curve by splitting it into **n strips of equal width** "
    "and replacing each strip with a **trapezoid**. Move the slider to see what happens as n grows."
)

try:
    expr = parse_function(func_text)
except Exception as err:  # noqa: BLE001
    st.error(f"Couldn't read that function. {err}")
    st.stop()

if b <= a:
    st.error("The upper limit b must be greater than the lower limit a.")
    st.stop()

f = sp.lambdify(x, expr, modules="numpy")
x_fine = np.linspace(a, b, 600)
y_fine = safe_eval(f, x_fine)

interval = sp.Interval(sp.nsimplify(a), sp.nsimplify(b))
try:
    from sympy.calculus.util import continuous_domain
    continuous = continuous_domain(expr, x, interval) == interval
except Exception:  # noqa: BLE001
    continuous = True  # fall back to the numerical check below

if not continuous or not np.all(np.isfinite(y_fine)):
    st.error(
        f"f(x) = {expr} isn't defined everywhere on [{fmt(a)}, {fmt(b)}] "
        "(for example, dividing by zero or taking the square root / log of a negative number). "
        "Choose a different interval."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------
h = (b - a) / n
xs = np.linspace(a, b, n + 1)
ys = safe_eval(f, xs)
approx = trapezoid(ys, h)

try:
    exact = float(sp.Integral(expr, (x, a, b)).evalf(15))
except Exception:  # noqa: BLE001
    exact = float(np.trapezoid(y_fine, x_fine)) if hasattr(np, "trapezoid") else float(np.trapz(y_fine, x_fine))

error = approx - exact
pct_error = abs(error / exact) * 100 if exact != 0 else float("nan")

# Concavity check: sign of f'' on the interval
f2 = sp.lambdify(x, sp.diff(expr, x, 2), modules="numpy")
f2_vals = safe_eval(f2, x_fine)
f2_vals = f2_vals[np.isfinite(f2_vals)]
tol = 1e-9
# f'' may touch 0 at isolated points (e.g. sin x at 0 and pi) without changing sign
if f2_vals.size and np.all(f2_vals > -tol) and np.any(f2_vals > tol):
    concavity = "up"
elif f2_vals.size and np.all(f2_vals < tol) and np.any(f2_vals < -tol):
    concavity = "down"
elif f2_vals.size and np.all(np.abs(f2_vals) <= tol):
    concavity = "linear"
else:
    concavity = "mixed"

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
st.latex(rf"\int_{{{fmt(a, 3)}}}^{{{fmt(b, 3)}}} {sp.latex(expr)}\,dx")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Trapezoidal estimate", fmt(approx))
m2.metric("Exact value", fmt(exact))
m3.metric("Error (estimate − exact)", fmt(error))
m4.metric("Percentage error", "—" if np.isnan(pct_error) else f"{pct_error:.4f} %")

tab_graph, tab_steps, tab_error, tab_predict = st.tabs(
    ["📈 Graph", "🧮 Step-by-step", "📉 How the error shrinks", "🤔 Predict"]
)

# ---------------------------------------------------------------------------
# Tab 1: graph
# ---------------------------------------------------------------------------
with tab_graph:
    fig = go.Figure()

    if show_area:
        fig.add_trace(go.Scatter(
            x=np.concatenate([[a], x_fine, [b]]), y=np.concatenate([[0], y_fine, [0]]),
            fill="toself", fillcolor="rgba(42,120,214,0.12)", line=dict(width=0),
            name="True area", hoverinfo="skip",
        ))

    for i in range(n):
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        strip_area = h / 2 * (y0 + y1)
        fig.add_trace(go.Scatter(
            x=[x0, x0, x1, x1, x0], y=[0, y0, y1, 0, 0],
            mode="lines", fill="toself", fillcolor="rgba(235,104,52,0.22)",
            line=dict(color=TRAP_COLOR, width=1.5),
            name="Trapezoids", legendgroup="traps", showlegend=(i == 0),
            hoveron="fills",
            text=f"Trapezoid {i + 1}<br>from x = {fmt(x0, 4)} to {fmt(x1, 4)}<br>"
                 f"area = (h/2)(y{i} + y{i + 1}) = {fmt(strip_area)}",
            hoverinfo="text",
        ))

    fig.add_trace(go.Scatter(
        x=x_fine, y=y_fine, mode="lines", line=dict(color=CURVE_COLOR, width=2.5),
        name=f"y = f(x)", hovertemplate="x = %{x:.4f}<br>f(x) = %{y:.4f}<extra></extra>",
    ))

    if show_points:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=9, color=TRAP_COLOR, line=dict(color="white", width=2)),
            name="(xᵢ, yᵢ)", customdata=np.arange(n + 1),
            hovertemplate="x<sub>%{customdata}</sub> = %{x:.4f}<br>y<sub>%{customdata}</sub> = %{y:.4f}<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color="gray", width=1))
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="x", yaxis_title="y", hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")

    # Over / under explanation
    if concavity == "up":
        st.info(
            "**Overestimate.** f(x) is **concave up** on this interval (f″(x) > 0), so the straight tops "
            "of the trapezoids sit *above* the curve. The estimate is larger than the true value."
        )
    elif concavity == "down":
        st.info(
            "**Underestimate.** f(x) is **concave down** on this interval (f″(x) < 0), so the straight tops "
            "of the trapezoids sit *below* the curve. The estimate is smaller than the true value."
        )
    elif concavity == "linear":
        st.success("f(x) is a straight line here, so the trapezoids fit it exactly: the error is zero.")
    else:
        st.warning(
            "f(x) changes concavity on this interval, so some trapezoids overestimate and others "
            "underestimate. You can't tell in advance which way the total error will go; "
            f"here the estimate is **{'too big' if error > 0 else 'too small'}**."
        )

    if (ys < 0).any():
        st.caption("Note: where the curve is below the x-axis, the area counts as negative.")

# ---------------------------------------------------------------------------
# Tab 2: step by step
# ---------------------------------------------------------------------------
with tab_steps:
    st.subheader("1. The formula")
    st.latex(r"\int_a^b f(x)\,dx \;\approx\; \frac{h}{2}\Big[y_0 + y_n + 2\,(y_1 + y_2 + \cdots + y_{n-1})\Big],"
             r"\qquad h = \frac{b-a}{n}")

    st.subheader("2. The strip width")
    a_tex = f"({fmt(a, 4)})" if a < 0 else fmt(a, 4)
    st.latex(rf"h = \frac{{{fmt(b, 4)} - {a_tex}}}{{{n}}} = {fmt(h)}")

    st.subheader("3. The y-values")
    weights = np.full(n + 1, 2)
    weights[0] = weights[-1] = 1
    table = pd.DataFrame({
        "i": np.arange(n + 1),
        "xᵢ": [fmt(v) for v in xs],
        "yᵢ = f(xᵢ)": [fmt(v) for v in ys],
        "weight": weights,
        "weight × yᵢ": [fmt(v) for v in weights * ys],
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("The two end values are used once; every inside value is shared by two trapezoids, so it's counted twice.")

    st.subheader("4. Substitute")
    ends = ys[0] + ys[-1]
    inner = ys[1:-1].sum()
    if n == 1:
        st.latex(rf"\frac{{{fmt(h)}}}{{2}}\Big[{fmt(ys[0])} + {fmt(ys[-1])}\Big] = {fmt(approx)}")
    else:
        inner_terms = " + ".join(fmt(v, 4) for v in ys[1:-1]) if n <= 7 else \
            f"{fmt(ys[1], 4)} + {fmt(ys[2], 4)} + \\cdots + {fmt(ys[-2], 4)}"
        st.latex(rf"\frac{{{fmt(h)}}}{{2}}\Big[{fmt(ys[0], 4)} + {fmt(ys[-1], 4)} + 2\,({inner_terms})\Big]")
        st.latex(rf"= \frac{{{fmt(h)}}}{{2}}\Big[{fmt(ends)} + 2 \times {fmt(inner)}\Big] = {fmt(approx)}")

    st.subheader("5. Check against the exact value")
    st.markdown(
        f"Exact value: **{fmt(exact)}**. Error = {fmt(approx)} − {fmt(exact)} = **{fmt(error)}**"
        + ("" if np.isnan(pct_error) else f", a percentage error of **{pct_error:.4f} %**.")
    )

# ---------------------------------------------------------------------------
# Tab 3: error vs n
# ---------------------------------------------------------------------------
with tab_error:
    ns = np.arange(1, 51)
    errs = []
    for k in ns:
        xk = np.linspace(a, b, k + 1)
        errs.append(abs(trapezoid(safe_eval(f, xk), (b - a) / k) - exact))
    errs = np.array(errs)

    plot_mask = errs > 1e-14
    if not plot_mask.any():
        st.success("The error is zero for every n here. That happens for straight lines, "
                   "and also when over- and underestimates cancel exactly (e.g. an odd function on a symmetric interval).")
    else:
        efig = go.Figure()
        efig.add_trace(go.Scatter(
            x=ns[plot_mask], y=errs[plot_mask], mode="lines+markers",
            line=dict(color=CURVE_COLOR, width=2), marker=dict(size=6),
            name="|error|", hovertemplate="n = %{x}<br>|error| = %{y:.3e}<extra></extra>",
        ))
        if errs[n - 1] > 1e-14:
            efig.add_trace(go.Scatter(
                x=[n], y=[errs[n - 1]], mode="markers",
                marker=dict(size=13, color=TRAP_COLOR, line=dict(color="white", width=2)),
                name="your n", hovertemplate="your n = %{x}<br>|error| = %{y:.3e}<extra></extra>",
            ))
        efig.update_layout(
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="n (number of trapezoids, log scale)", type="log",
                       tickvals=[1, 2, 5, 10, 20, 50], ticktext=["1", "2", "5", "10", "20", "50"]),
            yaxis=dict(title="|error| (log scale)", type="log", exponentformat="power", dtick=1),
            showlegend=False,
        )
        st.plotly_chart(efig, width="stretch")

        st.markdown(
            "On log-log axes the points fall close to a straight line with gradient about **−2**. "
            "That means the error is roughly proportional to **1/n²**: "
            "**doubling n cuts the error to about one quarter.**"
        )
        compare = pd.DataFrame({
            "n": [5, 10, 20, 40],
            "|error|": [f"{errs[k - 1]:.3e}" for k in (5, 10, 20, 40)],
        })
        compare["ratio to previous"] = ["—"] + [
            f"{errs[k1 - 1] / errs[k2 - 1]:.2f}" if errs[k2 - 1] > 1e-14 else "—"
            for k1, k2 in ((5, 10), (10, 20), (20, 40))
        ]
        st.dataframe(compare, hide_index=True)
        st.caption("Each ratio should be close to 4 for a smooth function.")

# ---------------------------------------------------------------------------
# Tab 4: predict
# ---------------------------------------------------------------------------
with tab_predict:
    st.markdown(
        "Before looking at the numbers, **predict**: will the trapezoidal rule give an "
        "overestimate or an underestimate for this function on this interval?"
    )
    guess = st.radio("My prediction", ["Overestimate", "Underestimate", "Can't tell"],
                     index=None, horizontal=True, key=f"guess_{choice}_{a}_{b}_{func_text}")
    if guess:
        actual = "Overestimate" if error > 1e-12 else "Underestimate" if error < -1e-12 else "Exact"
        if concavity == "mixed":
            correct = guess == "Can't tell" or guess == actual
            msg = (f"The function changes concavity here, so \"Can't tell\" is a fair answer from the graph alone. "
                   f"In fact the estimate turns out to be an **{actual.lower()}** (error = {fmt(error)}).")
        elif concavity == "linear":
            correct = False
            msg = "Trick question: it's a straight line, so the estimate is exact."
        else:
            correct = guess == actual
            reason = "concave up, so the trapezoid tops lie above the curve" if concavity == "up" \
                else "concave down, so the trapezoid tops lie below the curve"
            msg = f"It's an **{actual.lower()}**: the curve is {reason} (error = {fmt(error)})."
        (st.success if correct else st.error)(("✅ Correct! " if correct else "❌ Not quite. ") + msg)

st.markdown("---")
st.caption("Tip: hover over a trapezoid in the graph to see its individual area.")
