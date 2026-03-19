"""
Wafer Defect Analytics Dashboard — Phase 5
Plotly Dash web application for semiconductor material waste analysis
and predictive material requirement estimation.
Now using the Mixed-type Wafer Defect Dataset (38,015 wafers).
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Input, Output, State

# --- CONFIGURATION ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'wafer_control.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'material_model.pkl')

# Semiconductor-themed color palette
COLORS = {
    'bg': '#0a0e17',
    'card': '#131a2e',
    'card_border': '#1e2d4a',
    'accent': '#00d4ff',
    'accent2': '#7c3aed',
    'accent3': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'text': '#e2e8f0',
    'text_muted': '#94a3b8',
}

DEFECT_COLORS = {
    'Center': '#ef4444', 'Donut': '#f59e0b', 'Edge-Loc': '#10b981',
    'Edge-Ring': '#3b82f6', 'Loc': '#8b5cf6', 'Random': '#ec4899',
    'Scratch': '#06b6d4', 'Near-full': '#f97316', 'None': '#6b7280',
    'Undetected': '#374151',
}


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM wafer_logs", conn)
    conn.close()
    df['scan_time'] = pd.to_datetime(df['scan_time'])
    df['scan_date'] = df['scan_time'].dt.date
    return df


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None


# --- LOAD DATA ---
df = load_data()
model_pkg = load_model()

# --- PRE-COMPUTE STATS ---
total_scans = len(df)
fail_count = len(df[df['status'] == 'FAIL'])
pass_count = len(df[df['status'] == 'PASS'])
pass_rate = round((pass_count / total_scans) * 100, 1)
scrap_count = len(df[df['action'] == 'ROUTE_TO_SCRAP'])
total_waste = round(df['material_wasted_pct'].sum(), 1)
avg_waste = round(df[df['status'] == 'FAIL']['material_wasted_pct'].mean(), 2)
avg_confidence = round(df[df['status'] == 'FAIL']['confidence'].mean(), 2)

# Daily aggregations
daily = df.groupby('scan_date').agg(
    scans=('id', 'count'),
    fails=('status', lambda x: (x == 'FAIL').sum()),
    waste=('material_wasted_pct', lambda x: x.sum() / 100.0),
    avg_waste=('material_wasted_pct', 'mean'),
).reset_index()
daily['fail_rate'] = round((daily['fails'] / daily['scans']) * 100, 1)
daily['scan_date'] = pd.to_datetime(daily['scan_date'])

# ============================================================
# DASH APP
# ============================================================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Wafer Defect Analytics — Semiconductor QC Dashboard"

card_style = {
    'backgroundColor': COLORS['card'], 'border': f"1px solid {COLORS['card_border']}",
    'borderRadius': '12px', 'padding': '24px', 'marginBottom': '16px',
}

kpi_style = {
    'backgroundColor': COLORS['card'], 'border': f"1px solid {COLORS['card_border']}",
    'borderRadius': '12px', 'padding': '20px', 'textAlign': 'center', 'flex': '1', 'minWidth': '160px',
}


def make_kpi(title, value, subtitle="", color=COLORS['accent']):
    return html.Div(style=kpi_style, children=[
        html.P(title, style={'color': COLORS['text_muted'], 'fontSize': '12px', 'marginBottom': '4px', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.H2(str(value), style={'color': color, 'fontSize': '28px', 'fontWeight': '700', 'margin': '4px 0'}),
        html.P(subtitle, style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginTop': '4px'}),
    ])


def chart_layout(title):
    return dict(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=title, font=dict(size=16, color=COLORS['text'])),
        font=dict(color=COLORS['text_muted'], size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )


# ============================================================
# FIGURES
# ============================================================

# 1. Defect type distribution (pie) — YOLO predictions
defect_counts = df[df['status'] == 'FAIL']['defect_type'].value_counts().reset_index()
defect_counts.columns = ['defect_type', 'count']
fig_pie = px.pie(defect_counts, names='defect_type', values='count', color='defect_type',
                 color_discrete_map=DEFECT_COLORS, hole=0.45)
fig_pie.update_layout(**chart_layout('YOLOv8 Predicted Defect Distribution'))
fig_pie.update_traces(textinfo='label+percent', textfont_size=11)

# 2. Ground truth distribution (pie) — actual labels
gt_counts = df[df['status'] == 'FAIL']['ground_truth'].value_counts().reset_index()
gt_counts.columns = ['ground_truth', 'count']
fig_gt_pie = px.pie(gt_counts.head(15), names='ground_truth', values='count', hole=0.45)
fig_gt_pie.update_layout(**chart_layout('Ground Truth Label Distribution (Top 15)'))
fig_gt_pie.update_traces(textinfo='label+percent', textfont_size=10)

# 3. Material waste by defect type (bar)
waste_by_type = df[df['status'] == 'FAIL'].groupby('defect_type').agg(
    total_waste=('material_wasted_pct', lambda x: x.sum() / 100.0), count=('id', 'count'),
).reset_index().sort_values('total_waste', ascending=True)

fig_waste_bar = go.Figure()
fig_waste_bar.add_trace(go.Bar(
    y=waste_by_type['defect_type'], x=waste_by_type['total_waste'], orientation='h',
    marker_color=[DEFECT_COLORS.get(d, '#6b7280') for d in waste_by_type['defect_type']],
    text=[f"{v:.1f}" for v in waste_by_type['total_waste']], textposition='outside',
))
fig_waste_bar.update_layout(**chart_layout('Total Material Waste by Predicted Defect'))
fig_waste_bar.update_xaxes(title_text='Equivalent Lost Wafers')

# 4. Daily fail rate trend
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=daily['scan_date'], y=daily['fail_rate'], mode='lines+markers',
    line=dict(color=COLORS['danger'], width=2), marker=dict(size=5),
    name='Fail Rate %', fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)',
))
fig_trend.update_layout(**chart_layout('Daily Defect Rate Over Time'))
fig_trend.update_yaxes(title_text='Fail Rate %', range=[0, 105])

# 5. Daily waste trend
fig_waste_trend = go.Figure()
fig_waste_trend.add_trace(go.Scatter(
    x=daily['scan_date'], y=daily['waste'], mode='lines+markers',
    line=dict(color=COLORS['warning'], width=2), marker=dict(size=5),
    name='Lost Wafers', fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.1)',
))
fig_waste_trend.update_layout(**chart_layout('Daily Material Waste Over Time'))
fig_waste_trend.update_yaxes(title_text='Total Lost Wafers')

# 6. Action breakdown
action_counts = df['action'].value_counts().reset_index()
action_counts.columns = ['action', 'count']
action_colors = {'ROUTE_TO_SCRAP': COLORS['danger'], 'MOVE_TO_MICRO_STAGE': COLORS['warning'], 'ROUTE_TO_ASSEMBLY': COLORS['accent3']}
fig_action = px.bar(action_counts, x='action', y='count', color='action',
                    color_discrete_map=action_colors, text='count')
fig_action.update_layout(**chart_layout('Wafer Routing Actions'))
fig_action.update_traces(textposition='outside')

# 7. Feature importance
fig_importance = go.Figure()
if model_pkg:
    imp = model_pkg['metrics']['importances']
    imp_df = pd.DataFrame({'feature': list(imp.keys()), 'importance': list(imp.values())})
    imp_df = imp_df.sort_values('importance', ascending=True).tail(10)
    fig_importance.add_trace(go.Bar(
        y=imp_df['feature'], x=imp_df['importance'], orientation='h',
        marker_color=COLORS['accent2'],
        text=[f"{v:.3f}" for v in imp_df['importance']], textposition='outside',
    ))
fig_importance.update_layout(**chart_layout('Top 10 Prediction Features'))

# ============================================================
# LAYOUT
# ============================================================
app.layout = html.Div(style={
    'backgroundColor': COLORS['bg'], 'minHeight': '100vh',
    'fontFamily': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    'color': COLORS['text'], 'padding': '24px 32px',
}, children=[

    # HEADER
    html.Div(style={'marginBottom': '32px', 'borderBottom': f"1px solid {COLORS['card_border']}", 'paddingBottom': '20px'}, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '12px'}, children=[
            html.Div(style={'width': '12px', 'height': '12px', 'borderRadius': '50%',
                            'backgroundColor': COLORS['accent3'], 'boxShadow': f"0 0 8px {COLORS['accent3']}"}),
            html.H1("Wafer Defect Analytics", style={
                'margin': '0', 'fontSize': '28px', 'fontWeight': '700',
                'background': f"linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent2']})",
                'WebkitBackgroundClip': 'text', 'WebkitTextFillColor': 'transparent',
            }),
        ]),
        html.P("Mixed-type Wafer Defect Dataset — Material Waste Dashboard",
               style={'color': COLORS['text_muted'], 'marginTop': '4px', 'fontSize': '14px'}),
    ]),

    # TABS
    dcc.Tabs(id='tabs', value='tab-waste', style={'marginBottom': '24px'}, children=[
        dcc.Tab(label='📊 Historical Waste Analysis', value='tab-waste', style={
            'backgroundColor': COLORS['card'], 'color': COLORS['text_muted'],
            'border': f"1px solid {COLORS['card_border']}", 'borderRadius': '8px 8px 0 0',
            'padding': '12px 24px', 'fontWeight': '600',
        }, selected_style={
            'backgroundColor': COLORS['accent2'], 'color': '#fff',
            'border': f"1px solid {COLORS['accent2']}", 'borderRadius': '8px 8px 0 0',
            'padding': '12px 24px', 'fontWeight': '600',
        }),
        dcc.Tab(label='🔮 Material Prediction', value='tab-predict', style={
            'backgroundColor': COLORS['card'], 'color': COLORS['text_muted'],
            'border': f"1px solid {COLORS['card_border']}", 'borderRadius': '8px 8px 0 0',
            'padding': '12px 24px', 'fontWeight': '600',
        }, selected_style={
            'backgroundColor': COLORS['accent2'], 'color': '#fff',
            'border': f"1px solid {COLORS['accent2']}", 'borderRadius': '8px 8px 0 0',
            'padding': '12px 24px', 'fontWeight': '600',
        }),
    ]),

    html.Div(id='tab-content'),
])


# ============================================================
# CALLBACKS
# ============================================================
@callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-waste':
        return html.Div([
            # KPI Row
            html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '24px'}, children=[
                make_kpi("Total Scans", f"{total_scans:,}", "wafers inspected", COLORS['accent']),
                make_kpi("Pass Rate", f"{pass_rate}%", f"{pass_count:,} passed", COLORS['accent3']),
                make_kpi("Fail Rate", f"{100-pass_rate}%", f"{fail_count:,} defective", COLORS['danger']),
                make_kpi("Scrapped", f"{scrap_count:,}", "routed to scrap", COLORS['warning']),
                make_kpi("Avg Waste/Wafer", f"{avg_waste}%", "per defective wafer", COLORS['danger']),
                make_kpi("Avg Confidence", f"{avg_confidence}", "model certainty", COLORS['accent3']),
            ]),

            # Charts Row 1: YOLO predictions vs Ground Truth
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px', 'marginBottom': '16px'}, children=[
                html.Div(style=card_style, children=[dcc.Graph(figure=fig_pie, config={'displayModeBar': False})]),
                html.Div(style=card_style, children=[dcc.Graph(figure=fig_gt_pie, config={'displayModeBar': False})]),
            ]),

            # Charts Row 2
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px', 'marginBottom': '16px'}, children=[
                html.Div(style=card_style, children=[dcc.Graph(figure=fig_waste_bar, config={'displayModeBar': False})]),
                html.Div(style=card_style, children=[dcc.Graph(figure=fig_action, config={'displayModeBar': False})]),
            ]),

            # Trend charts
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px'}, children=[
                html.Div(style=card_style, children=[dcc.Graph(figure=fig_trend, config={'displayModeBar': False})]),
                html.Div(style=card_style, children=[dcc.Graph(figure=fig_waste_trend, config={'displayModeBar': False})]),
            ]),
        ])

    elif tab == 'tab-predict':
        model_status = "✅ Model loaded" if model_pkg else "❌ No model found"
        model_metrics = ""
        if model_pkg:
            m = model_pkg['metrics']
            model_metrics = f"R² = {m['r2']:.4f}  |  MAE = {m['mae']:.2f}%"

        return html.Div([
            html.Div(style={**card_style, 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
                html.Div([
                    html.H3("Prediction Model", style={'margin': '0 0 4px 0', 'fontSize': '18px'}),
                    html.P(model_status, style={'margin': '0', 'color': COLORS['accent3'] if model_pkg else COLORS['danger']}),
                ]),
                html.P(model_metrics, style={'color': COLORS['text_muted'], 'fontFamily': 'monospace'}),
            ]),

            html.Div(style=card_style, children=[
                html.H3("Forecast Parameters", style={'marginTop': '0', 'fontSize': '18px', 'marginBottom': '20px'}),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'}, children=[
                    html.Div([
                        html.Label("Expected Daily Production (wafers)", style={'color': COLORS['text_muted'], 'fontSize': '13px'}),
                        dcc.Slider(id='slider-scans', min=100, max=2000, step=50, value=1300,
                                   marks={100: '100', 500: '500', 1000: '1000', 1500: '1500', 2000: '2000'},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ]),
                    html.Div([
                        html.Label("Expected Defect Rate (%)", style={'color': COLORS['text_muted'], 'fontSize': '13px'}),
                        dcc.Slider(id='slider-fail-rate', min=0, max=100, step=5, value=97,
                                   marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ]),
                ]),
                html.Br(),
                html.Button("🔮 Predict Material Needs", id='btn-predict', n_clicks=0, style={
                    'backgroundColor': COLORS['accent2'], 'color': '#fff', 'border': 'none',
                    'borderRadius': '8px', 'padding': '12px 32px', 'fontSize': '15px',
                    'fontWeight': '600', 'cursor': 'pointer', 'width': '100%',
                }),
            ]),

            html.Div(id='prediction-result'),

            html.Div(style=card_style, children=[
                dcc.Graph(figure=fig_importance, config={'displayModeBar': False}),
            ]),
        ])


@callback(
    Output('prediction-result', 'children'),
    Input('btn-predict', 'n_clicks'),
    State('slider-scans', 'value'),
    State('slider-fail-rate', 'value'),
    prevent_initial_call=True,
)
def predict(n_clicks, n_scans, fail_pct):
    if not model_pkg:
        return html.Div(style={**card_style, 'borderColor': COLORS['danger']}, children=[
            html.P("❌ No model loaded.", style={'color': COLORS['danger']}),
        ])

    from material_predictor import predict_material_needs
    model = model_pkg['model']
    feat_cols = model_pkg['feature_cols']

    fail_df = df[df['status'] == 'FAIL']
    dist = fail_df['defect_type'].value_counts(normalize=True).to_dict()

    pred = predict_material_needs(model, feat_cols, n_scans, fail_pct / 100.0, dist)

    return html.Div(style={
        **card_style, 'borderColor': COLORS['accent'],
        'background': f"linear-gradient(135deg, {COLORS['card']}, #1a1040)",
    }, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'gap': '16px'}, children=[
            make_kpi("Daily Production", f"{n_scans}", "wafers", COLORS['accent']),
            make_kpi("Expected Defect Rate", f"{fail_pct}%", f"~{int(n_scans * fail_pct / 100)} defective", COLORS['danger']),
            make_kpi("Avg Waste/Wafer", f"{pred['avg_waste_per_wafer']:.1f}%", "per defective wafer loss", COLORS['warning']),
            make_kpi("Total Daily Waste", f"{pred['total_daily_waste']:.1f} wafers", "total estimated loss", COLORS['danger']),
        ]),
    ])


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  WAFER DEFECT ANALYTICS DASHBOARD")
    print(f"  Data: {total_scans:,} scans | {fail_count:,} defects | {pass_count:,} pass")
    print(f"  Model: {'Loaded ✅' if model_pkg else 'Not found ❌'}")
    print("=" * 60)
    print(f"\n  🌐 Open: http://127.0.0.1:8050\n")
    app.run(debug=True, port=8050)
