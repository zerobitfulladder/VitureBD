"""menu.py — the settings menu's model (no GL, no input handling).

Settings are grouped into Tabs. Within a tab, items are either a Section (a
non-selectable subheader) or a Setting (a typed value: num / bool / enum, with
get/set closures over the app's state, or an action). Settings and tabs may
carry a `visible` predicate so context-dependent knobs (e.g. JBU vs guided-
filter options, or everything that only means something in 3D) appear only
when relevant.

It is driven by row: the session maps the pointer to a row of rows() and
hovers it (the selected row), clicks it, or steps it up and down.
"""


class Section:
    """A non-selectable subheader that groups the settings beneath it."""
    is_section = True

    def __init__(self, label):
        self.label = label


class Setting:
    is_section = False

    def __init__(self, label, kind, get, set, *, step=1, lo=None, hi=None,
                 options=None, fmt=None, editable=True, visible=None, on_click=None,
                 help=""):
        self.label = label
        self.kind = kind            # 'num' | 'bool' | 'enum' | 'action'
        self._get, self._set = get, set
        self.step, self.lo, self.hi = step, lo, hi
        self.options = options or []
        self._fmt = fmt
        self.editable = editable
        self._visible = visible     # None => always visible
        self.on_click = on_click    # replaces the click's toggle / next (apply, act)
        self.help = help

    @property
    def visible(self):
        return self._visible is None or bool(self._visible())

    @property
    def value(self):
        return self._get()

    def display(self):
        v = self._get()
        if self._fmt:
            return self._fmt(v)
        if self.kind == "bool":
            return "ON" if v else "OFF"
        if self.kind == "enum":
            return str(v)
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    def click(self):
        """Its on_click if it has one; else a bool toggles and an enum steps on."""
        if self.on_click is not None:
            self.on_click()
        elif self.kind in ("bool", "enum"):
            self.adjust(+1)

    def adjust(self, direction):
        """One step: -1 down, +1 up. Returns True if the value changed."""
        if not self.editable or self.kind == "action":
            return False
        v = self._get()
        if self.kind == "bool":
            nv = not v
        elif self.kind == "enum":
            i = self.options.index(v) if v in self.options else 0
            nv = self.options[(i + direction) % len(self.options)]
        else:  # num
            nv = v + direction * self.step
            if self.lo is not None:
                nv = max(self.lo, nv)
            if self.hi is not None:
                nv = min(self.hi, nv)
            if isinstance(v, int) and not isinstance(self.step, float):
                nv = int(round(nv))
        if nv == v:
            return False
        self._set(nv)
        return True


class Tab:
    def __init__(self, name, items, visible=None):
        self.name = name
        self.items = items          # Section | Setting, in display order
        self._visible = visible     # None => always visible

    @property
    def visible(self):
        return self._visible is None or bool(self._visible())


class Menu:
    def __init__(self, tabs):
        self.tabs = tabs
        self.t = 0                  # current tab index
        self._sel = 0               # selection within the current tab's nav list

    def _tabs(self):
        return [tab for tab in self.tabs if tab.visible]

    @property
    def tab(self):
        tabs = self._tabs()
        self.t = min(self.t, len(tabs) - 1)        # clamp if tabs hid out
        return tabs[self.t]

    def _nav(self):
        """Visible, editable settings in the current tab — where the cursor lands."""
        return [it for it in self.tab.items
                if not it.is_section and it.editable and it.visible]

    @property
    def current(self):
        nav = self._nav()
        if not nav:
            return None
        self._sel = min(self._sel, len(nav) - 1)   # clamp if rows hid out
        return nav[self._sel]

    def next_tab(self):
        self.t = (self.t + 1) % len(self._tabs())
        self._sel = 0

    def prev_tab(self):
        self.t = (self.t - 1) % len(self._tabs())
        self._sel = 0

    def select_tab(self, i):
        self.t = i % len(self._tabs())
        self._sel = 0

    def _row_items(self):
        """The items rows() shows, in order."""
        return [it for it in self.tab.items if it.is_section or it.visible]

    def _row(self, i):
        """rows()[i]'s setting, selected; None for a section."""
        items = self._row_items()
        it = items[i] if 0 <= i < len(items) else None
        if it is None or it.is_section or not it.editable:
            return None
        self._sel = self._nav().index(it)
        return it

    def hover(self, i):
        self._row(i)

    def click(self, i):
        it = self._row(i)
        if it is not None:
            it.click()

    def step(self, i, direction):
        it = self._row(i)
        if it is not None:
            it.adjust(direction)

    def tab_bar(self):
        """(names, current_index) for the panel's tab strip."""
        tab = self.tab
        return [t.name for t in self._tabs()], self._tabs().index(tab)

    def rows(self):
        """Rows of the CURRENT tab only (sections + visible settings)."""
        cur = self.current
        out = []
        for it in self.tab.items:
            if it.is_section:
                out.append({"section": True, "label": it.label})
            elif it.visible:
                out.append({"section": False, "label": it.label, "value": it.display(),
                            "kind": it.kind, "editable": it.editable,
                            "selected": (it is cur), "help": it.help})
        return out
