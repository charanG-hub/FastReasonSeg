from pathlib import Path

path = Path("app.py")
source = path.read_text()

old = '''            # Default to the highest-confidence candidate.
            selected = st.multiselect(
                "Choose candidates to segment",
                options=list(range(len(boxes))),
                default=[0],
                format_func=lambda i: detection_options[i],
                key="multi_candidate_selection",
                help="Select one candidate, several candidates, or all candidates.",
            )

            button_col1, button_col2 = st.columns(2)
            with button_col1:
                if st.button("Select all", key="select_all_candidates"):
                    st.session_state["multi_candidate_selection"] = list(range(len(boxes)))
                    st.rerun()

            with button_col2:
                if st.button("Clear selection", key="clear_candidate_selection"):
                    st.session_state["multi_candidate_selection"] = []
                    st.rerun()
'''

new = '''            # Initialize selection before creating the multiselect widget.
            if "multi_candidate_selection" not in st.session_state:
                st.session_state["multi_candidate_selection"] = [0]

            def select_all_candidates():
                current_boxes = st.session_state.get("multi_candidate_boxes", [])
                st.session_state["multi_candidate_selection"] = list(
                    range(len(current_boxes))
                )

            def clear_candidate_selection():
                st.session_state["multi_candidate_selection"] = []

            button_col1, button_col2 = st.columns(2)
            with button_col1:
                st.button(
                    "Select all",
                    key="select_all_candidates",
                    on_click=select_all_candidates,
                )

            with button_col2:
                st.button(
                    "Clear selection",
                    key="clear_candidate_selection",
                    on_click=clear_candidate_selection,
                )

            selected = st.multiselect(
                "Choose candidates to segment",
                options=list(range(len(boxes))),
                format_func=lambda i: detection_options[i],
                key="multi_candidate_selection",
                help="Select one candidate, several candidates, or all candidates.",
            )
'''

if old not in source:
    print("ERROR: The expected code block was not found. No changes made.")
    print("Keep app_before_selection_fix.py as your backup.")
    raise SystemExit(1)

path.write_text(source.replace(old, new, 1))
print("Selection fix applied successfully.")
