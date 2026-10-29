from pathlib import Path
import shutil

path = Path("app.py")
source = path.read_text()

start_marker = '                            with st.form(key=f"refine_form_{index}"):'
end_marker = '                                try:\n                                    with st.spinner('

start = source.find(start_marker)
end = source.find(end_marker, start)

if start < 0 or end < 0:
    raise SystemExit(
        "Could not find the expected refinement form. "
        "No changes made."
    )

replacement = '''                            with st.form(key=f"refine_form_{index}"):
                                st.markdown("**Positive points — keep this person**")
                                positive_points_text = st.text_area(
                                    "Positive coordinates (x, y), one point per line",
                                    value=(
                                        f"{int((boxes[index][0] + boxes[index][2]) / 2)}, "
                                        f"{int((boxes[index][1] + boxes[index][3]) / 2)}"
                                    ),
                                    height=100,
                                    key=f"positive_points_text_{index}",
                                    help="Example: 900, 300",
                                )

                                st.markdown("**Negative points — exclude other people**")
                                negative_points_text = st.text_area(
                                    "Negative coordinates (x, y), one point per line",
                                    value="",
                                    height=100,
                                    key=f"negative_points_text_{index}",
                                    help="Example: 1050, 350",
                                )

                                refine_clicked = st.form_submit_button(
                                    "Refine mask with points",
                                    type="primary",
                                )

                            if refine_clicked:
                                point_coords = []
                                point_labels = []
                                invalid_lines = []

                                def parse_point_lines(text, label):
                                    parsed = []
                                    bad_lines = []

                                    for line_number, line in enumerate(
                                        text.splitlines(), start=1
                                    ):
                                        line = line.strip()
                                        if not line:
                                            continue

                                        parts = line.replace(";", ",").split(",")
                                        if len(parts) != 2:
                                            bad_lines.append(line_number)
                                            continue

                                        try:
                                            x = int(parts[0].strip())
                                            y = int(parts[1].strip())
                                        except ValueError:
                                            bad_lines.append(line_number)
                                            continue

                                        if not (
                                            0 <= x < saved_image.width
                                            and 0 <= y < saved_image.height
                                        ):
                                            bad_lines.append(line_number)
                                            continue

                                        parsed.append((x, y, label))

                                    return parsed, bad_lines

                                positive_points, bad_positive = parse_point_lines(
                                    positive_points_text, 1
                                )
                                negative_points, bad_negative = parse_point_lines(
                                    negative_points_text, 0
                                )
                                invalid_lines = bad_positive + bad_negative

                                if invalid_lines:
                                    st.warning(
                                        "Some coordinate lines were ignored. "
                                        "Use valid integer x,y coordinates within "
                                        "the original image dimensions."
                                    )

                                if not positive_points:
                                    st.error(
                                        "Enter at least one valid positive point."
                                    )
                                else:
                                    all_points = positive_points + negative_points
                                    point_coords = [
                                        [float(x), float(y)]
                                        for x, y, _ in all_points
                                    ]
                                    point_labels = [
                                        label for _, _, label in all_points
                                    ]

'''

backup = path.with_name("app_before_multiple_points.py")
shutil.copy2(path, backup)
path.write_text(source[:start] + replacement + source[end:])

print("Added multiple positive and negative point inputs.")
print(f"Backup: {backup}")
