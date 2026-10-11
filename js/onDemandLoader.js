import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { SearchableCombo } from "./SearchableCombo.js";

async function onLoraChanged(node, lora_name) {
    try {
        const response = await api.fetchApi("/on_demand_loader/lora_changed", {
            method: "POST",
            body: JSON.stringify({ lora_name }),
            headers: {
                "Content-Type": "application/json",
            },
        });
        if (response.status !== 200) {
            console.error(`[ComfyUI-OnDemand-Loaders] Failed to notify backend of lora change: ${response.status}`);
        } else {
			const loraInfo = await response.json();
			addOrUpdateLoraInfoWidgets(node, loraInfo);
		}
    } catch (e) {
        console.error("[ComfyUI-OnDemand-Loaders] Failed to notify backend of lora change", e);
    }
}

async function onFilterChanged(node, filter) {
    try {
        const response = await api.fetchApi("/on_demand_loader/filter_changed", {
            method: "POST",
            body: JSON.stringify({ filter }),
            headers: {
                "Content-Type": "application/json",
            },
        });
        if (response.status !== 200) {
            console.error(`[ComfyUI-OnDemand-Loaders] Failed to notify backend of filter change: ${response.status}`);
        } else {
			const filteredLoras = await response.json();
			const w = node.widgets.find((w) => w.name === "lora_name")
			w.options.values = filteredLoras.map(lora => lora.name);
		}
    } catch (e) {
        console.error("[ComfyUI-OnDemand-Loaders] Failed to notify backend of filter change", e);
    }
}

function addOrUpdateLoraInfoWidgets(node, loraInfo) {
	if (loraInfo.name !== "None") {

		if (loraInfo.author) {
			addOrUpdateWidget(node, "author", loraInfo.author);
		}

		if (loraInfo.trigger_words && loraInfo.trigger_words.length > 0) {
			addOrUpdateWidget(node, "trigger_words", loraInfo.trigger_words.join(', '), "STRING", true);
		}

		if (loraInfo.id) {
			const urlValue = `[Civitai Model URL](https://civitai.com/models/${loraInfo.id})`;
			addOrUpdateWidget(node, "url", urlValue, "MARKDOWN", false);
		}

		if (loraInfo.base_model) {
			addOrUpdateWidget(node, "base_model", loraInfo.base_model);
		}
	} else {
		hideWidgetIfExists(node, "author");
		hideWidgetIfExists(node, "trigger_words");
		hideWidgetIfExists(node, "url");
		hideWidgetIfExists(node, "base_model");
	}
	node.computeSize();
	if (typeof node.setDirtyCanvas === 'function') {
      node.setDirtyCanvas(true, true);
    }
}

function hideWidgetIfExists(node, widgetName) {
	const w = node.widgets.find((w) => w.name === widgetName);
	if (w) {
		w.hidden = true;
	}
}

function addOrUpdateWidget(node, widgetName, value, type = "STRING", multiline = false) {
	const w = node.widgets.find((w) => w.name === widgetName);
	if (!w) {
		const newW = ComfyWidgets[type](node, widgetName, ["STRING", { multiline: multiline }], app).widget;
		if (newW.inputEl) {
			newW.inputEl.readOnly = true;
			newW.inputEl.style.opacity = 0.8;
		}
		newW.value = value;
	} else {
		w.value = value
		w.hidden = false;
	}
}

// Config-based loaders: each `url` widget is tied to a model name widget and a config.json list
// through the ondemand_* options of its input definition (see _url_input in nodes.py).
function getUrlLinks(nodeData) {
	return Object.entries(nodeData.input?.optional ?? {})
		.filter(([, spec]) => spec?.[1]?.ondemand_config_key)
		.map(([urlInput, spec]) => ({
			urlInput,
			nameInput: spec[1].ondemand_name_input,
			configKey: spec[1].ondemand_config_key,
		}));
}

function getLinkWidgets(node, link) {
	return {
		nameWidget: node.widgets?.find((w) => w.name === link.nameInput),
		urlWidget: node.widgets?.find((w) => w.name === link.urlInput),
	};
}

async function populateUrl(node, link, { onlyIfEmpty = false } = {}) {
	const { nameWidget, urlWidget } = getLinkWidgets(node, link);
	if (!nameWidget || !urlWidget) return;
	const name = nameWidget.value;
	try {
		const params = new URLSearchParams({ key: link.configKey, name });
		const response = await api.fetchApi(`/on_demand_loader/model_url?${params}`);
		if (response.status !== 200) {
			console.error(`[ComfyUI-OnDemand-Loaders] Failed to fetch url for '${name}': ${response.status}`);
			return;
		}
		const { url } = await response.json();
		// Ignore stale responses, and keep URLs already saved in a loaded workflow
		if (nameWidget.value !== name || (onlyIfEmpty && urlWidget.value)) return;
		// A name that is not in config.json yet keeps the URL the user entered
		if (url || name === "None") {
			urlWidget.value = url;
			node.setDirtyCanvas?.(true, true);
		}
	} catch (e) {
		console.error(`[ComfyUI-OnDemand-Loaders] Failed to fetch url for '${name}'`, e);
	}
}

// The values array is shared by all nodes of a type; `local` adds a value to this widget only,
// for names that are not saved in config.json yet.
function addComboValue(widget, value, { local = false } = {}) {
	const values = widget.options?.values;
	if (!Array.isArray(values) || values.includes(value)) return;
	if (local) {
		widget.options = { ...widget.options, values: [...values, value] };
	} else {
		values.push(value);
	}
}

async function promptNewModelName(node, link) {
	const { nameWidget, urlWidget } = getLinkWidgets(node, link);
	if (!nameWidget || !urlWidget) return;

	// The backend checks the name, or gets it from the server when it is left blank
	const apiKey = node.widgets?.find((w) => w.name === "api_key")?.value || "";
	const check = async (url, subfolder, name) => {
		const response = await api.fetchApi("/on_demand_loader/new_model", {
			method: "POST",
			body: JSON.stringify({ key: link.configKey, url, subfolder, name, api_key: apiKey }),
			headers: { "Content-Type": "application/json" },
		});
		return response.json();
	};
	// Existing subfolders of this model type's folder, offered as suggestions
	const listSubfolders = async () => {
		const response = await api.fetchApi(`/on_demand_loader/subfolders?${new URLSearchParams({ key: link.configKey })}`);
		return response.status === 200 ? (await response.json()).subfolders : [];
	};

	const result = await showNewModelDialog(`New ${link.nameInput}`, check, listSubfolders);
	if (!result) return;

	addComboValue(nameWidget, result.name, { local: true });
	nameWidget.value = result.name;
	urlWidget.value = result.url;
	node.setDirtyCanvas?.(true, true);
}

// A <dialog> with this extension's styling, from HTML whose .ondemand-title element receives title. Not yet shown.
function makeDialog(title, html) {
	const dialog = document.createElement("dialog");
	dialog.style.cssText = `
		padding: 20px; width: min(520px, calc(100vw - 32px)); box-sizing: border-box;
		border: 1px solid var(--border-color, #444); border-radius: 8px;
		background: var(--comfy-menu-bg, #202020); color: var(--fg-color, #ddd);
		font: 14px sans-serif;`;
	dialog.innerHTML = html;
	dialog.querySelector(".ondemand-title").textContent = title;
	for (const el of dialog.querySelectorAll("input:not([type=radio]), button")) {
		el.style.cssText = `
			padding: 6px 10px; border-radius: 6px; font: inherit;
			border: 1px solid var(--border-color, #444);
			background: var(--comfy-input-bg, #2a2a2a); color: var(--input-text, #ddd);`;
	}
	return dialog;
}

// Shows a dialog from makeDialog as a modal and returns a promise for its result. setup(finish, isFinished) wires up the
// dialog; finish(value) closes it and resolves the promise with value. Escape resolves to null. The result does not
// wait for the dialog's close event, which browsers may hold back while the page is not being rendered.
function runDialog(dialog, setup) {
	return new Promise((resolve) => {
		let finished = false;
		const finish = (value) => {
			if (finished) return;
			finished = true;
			if (dialog.open) dialog.close();
			dialog.remove();
			resolve(value);
		};
		dialog.addEventListener("cancel", (e) => {
			e.preventDefault();
			finish(null);
		});
		dialog.addEventListener("close", () => finish(null));
		document.body.appendChild(dialog);
		dialog.showModal();
		setup(finish, () => finished);
	});
}

// Modal asking for a download URL, optional subfolder and optional model name. check(url, subfolder, name) resolves to
// { name } (the full name, including the subfolder) or { error }; listSubfolders() resolves to existing subfolders.
// The dialog resolves to { name, url }, or null if cancelled.
function showNewModelDialog(title, check, listSubfolders) {
	const dialog = makeDialog(title, `
			<form method="dialog" style="display: flex; flex-direction: column; gap: 12px; margin: 0;">
				<div class="ondemand-title" style="font-size: 16px; font-weight: 600;"></div>
				<div style="color: var(--descrip-text, #999);">The model is added to config.json and downloaded when the workflow runs.</div>
				<label style="display: flex; flex-direction: column; gap: 4px;">URL
					<input name="url" required autocomplete="off" placeholder="https://civitai.com/api/download/models/... or https://huggingface.co/.../resolve/...">
				</label>
				<label style="display: flex; flex-direction: column; gap: 4px;">
					<span>Subfolder <span style="color: var(--descrip-text, #999);">(optional)</span></span>
					<input name="subfolder" autocomplete="off" list="ondemand-subfolders" placeholder="For example zimage/characters">
					<datalist id="ondemand-subfolders"></datalist>
				</label>
				<label style="display: flex; flex-direction: column; gap: 4px;">
					<span>Name <span style="color: var(--descrip-text, #999);">(optional)</span></span>
					<input name="name" autocomplete="off" placeholder="Leave blank to use the file name from the server">
				</label>
				<div class="ondemand-error" role="alert" style="color: var(--error-text, #ff6b6b); white-space: pre-wrap;" hidden></div>
				<div style="display: flex; justify-content: flex-end; gap: 8px;">
					<button type="button" value="cancel">Cancel</button>
					<button type="submit" value="ok">Add model</button>
				</div>
			</form>`);

	return runDialog(dialog, (finish, isFinished) => {
		const form = dialog.querySelector("form");
		const errorEl = dialog.querySelector(".ondemand-error");
		const submitButton = dialog.querySelector('button[type="submit"]');
		const showError = (message) => {
			errorEl.textContent = message;
			errorEl.hidden = !message;
		};

		form.addEventListener("submit", async (e) => {
			e.preventDefault();
			const url = form.elements.url.value.trim();
			const subfolder = form.elements.subfolder.value.trim();
			const name = form.elements.name.value.trim();
			showError("");
			form.querySelectorAll("input, button").forEach((el) => (el.disabled = true));
			submitButton.textContent = name ? "Checking…" : "Getting file name…";
			let checked;
			try {
				checked = await check(url, subfolder, name);
			} catch (err) {
				checked = { error: `Could not check the model: ${err}` };
			}
			if (isFinished()) return; // Cancelled while checking
			form.querySelectorAll("input, button").forEach((el) => (el.disabled = false));
			submitButton.textContent = "Add model";
			if (checked.error) {
				showError(checked.error);
				return;
			}
			finish({ name: checked.name, url });
		});
		dialog.querySelector('button[value="cancel"]').addEventListener("click", () => finish(null));
		// Whitespace-only URLs pass `required`; reject them like empty ones
		const urlInput = form.elements.url;
		urlInput.addEventListener("input", () => urlInput.setCustomValidity(urlInput.value.trim() ? "" : "Required"));

		listSubfolders()
			.then((subfolders) => {
				const datalist = dialog.querySelector("datalist");
				for (const subfolder of subfolders) {
					const option = document.createElement("option");
					option.value = subfolder;
					datalist.appendChild(option);
				}
			})
			.catch((err) => console.error("[ComfyUI-OnDemand-Loaders] Failed to list subfolders", err));

		urlInput.focus();
	});
}

function setupUrlWidgets(nodeType, links) {
	const onNodeCreated = nodeType.prototype.onNodeCreated;
	nodeType.prototype.onNodeCreated = function () {
		const r = onNodeCreated?.apply(this, arguments);
		for (const link of links) {
			const { nameWidget } = getLinkWidgets(this, link);
			if (!nameWidget) continue;

			const callback = nameWidget.callback;
			const node = this;
			nameWidget.callback = function () {
				const r = callback?.apply(this, arguments);
				populateUrl(node, link);
				return r;
			};

			const label = links.length > 1 ? `➕ New ${link.nameInput}` : "➕ New model";
			this.addWidget("button", label, "", () => promptNewModelName(this, link), { serialize: false });

			populateUrl(this, link, { onlyIfEmpty: true });
		}
		return r;
	};

	const onConfigure = nodeType.prototype.onConfigure;
	nodeType.prototype.onConfigure = function () {
		const r = onConfigure?.apply(this, arguments);
		for (const link of links) {
			const { nameWidget } = getLinkWidgets(this, link);
			// Names saved in the workflow but not in config.json yet must stay selectable
			if (nameWidget) addComboValue(nameWidget, nameWidget.value, { local: true });
			populateUrl(this, link, { onlyIfEmpty: true });
		}
		return r;
	};
}

// Exporting and importing config.json, from the node's right-click menu

const SECTION_LABELS = {
	loras: "LoRA",
	diffusion_models: "diffusion",
	checkpoints: "checkpoint",
	vae_models: "VAE",
	clip_models: "CLIP",
	clip_vision: "CLIP Vision",
	gguf_models: "GGUF",
	controlnet_models: "ControlNet",
	model_patches: "model patch",
};

function notify(severity, summary, detail) {
	const toast = app.extensionManager?.toast;
	if (toast?.add) {
		toast.add({ severity, summary, detail, life: severity === "error" ? 8000 : 4000 });
	} else {
		(severity === "error" ? console.error : console.log)(`[ComfyUI-OnDemand-Loaders] ${summary}: ${detail}`);
	}
}

// Saves the given config.json sections (all of them if keys is null) as a file through the browser
async function exportModels(keys) {
	try {
		const params = keys ? `?${new URLSearchParams({ keys: keys.join(",") })}` : "";
		const response = await api.fetchApi(`/on_demand_loader/export${params}`);
		const data = await response.json();
		if (response.status !== 200) throw new Error(data.error ?? `HTTP ${response.status}`);

		const blob = new Blob([JSON.stringify(data, null, 4) + "\n"], { type: "application/json" });
		const link = document.createElement("a");
		link.href = URL.createObjectURL(blob);
		link.download = keys?.length === 1 ? `ondemand-${keys[0]}.json` : "ondemand-models.json";
		link.click();
		setTimeout(() => URL.revokeObjectURL(link.href), 1000);
	} catch (e) {
		notify("error", "Export failed", String(e.message ?? e));
	}
}

// Lets the user pick an exported file, shows what importing it would change, and merges it into config.json
function importModels() {
	const input = document.createElement("input");
	input.type = "file";
	input.accept = ".json,application/json";
	input.addEventListener("change", async () => {
		const file = input.files?.[0];
		if (!file) return;
		let config;
		try {
			config = JSON.parse(await file.text());
		} catch (e) {
			notify("error", "Import failed", `${file.name} is not a JSON file: ${e.message}`);
			return;
		}
		const post = async (apply, useImported) => {
			const response = await api.fetchApi("/on_demand_loader/import", {
				method: "POST",
				body: JSON.stringify({ config, apply, use_imported: useImported }),
				headers: { "Content-Type": "application/json" },
			});
			const data = await response.json();
			if (response.status !== 200) throw new Error(data.error ?? `HTTP ${response.status}`);
			return data;
		};
		try {
			const preview = await post(false, false);
			const useImported = await showImportDialog(file.name, preview);
			if (useImported === null) return;
			const result = await post(true, useImported);
			if (result.applied) {
				notify("success", "Models imported", `${result.applied} model${result.applied === 1 ? "" : "s"} added or updated in config.json`);
				await app.refreshComboInNodes?.();
			}
		} catch (e) {
			notify("error", "Import failed", String(e.message ?? e));
		}
	});
	input.click();
}

// Shows what importing would change; resolves to whether conflicts should use the imported entries, or null if cancelled
function showImportDialog(fileName, preview) {
	const dialog = makeDialog(`Import models from ${fileName}`, `
			<form method="dialog" style="display: flex; flex-direction: column; gap: 12px; margin: 0;">
				<div class="ondemand-title" style="font-size: 16px; font-weight: 600; overflow-wrap: anywhere;"></div>
				<div class="ondemand-summary" style="display: flex; flex-direction: column; gap: 10px; max-height: 50vh; overflow-y: auto;"></div>
				<fieldset class="ondemand-conflicts" style="border: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 4px;" hidden>
					<legend style="padding: 0; margin-bottom: 4px;">For models in both, with different URLs:</legend>
					<label><input type="radio" name="conflicts" value="keep" checked> Keep mine</label>
					<label><input type="radio" name="conflicts" value="imported"> Use imported</label>
				</fieldset>
				<div class="ondemand-nothing" style="color: var(--descrip-text, #999);" hidden>Nothing would be added or changed.</div>
				<div style="display: flex; justify-content: flex-end; gap: 8px;">
					<button type="button" value="cancel">Cancel</button>
					<button type="submit" value="ok">Import</button>
				</div>
			</form>`);

	return runDialog(dialog, (finish) => {
		const summary = dialog.querySelector(".ondemand-summary");
		const muted = "color: var(--descrip-text, #999);";
		const addLine = (parent, text, style = "") => {
			const line = document.createElement("div");
			line.textContent = text;
			line.style.cssText = style;
			parent.appendChild(line);
			return line;
		};
		const addList = (parent, items) => {
			const list = document.createElement("ul");
			list.style.cssText = `margin: 2px 0 0; padding-left: 20px; ${muted} overflow-wrap: anywhere;`;
			for (const item of items) {
				const li = document.createElement("li");
				li.textContent = item;
				list.appendChild(li);
			}
			parent.appendChild(list);
		};

		let conflictCount = 0;
		for (const [key, section] of Object.entries(preview.sections)) {
			const block = document.createElement("div");
			const counts = [
				`${section.added.length} new`,
				`${section.unchanged.length + section.hash_added.length} already present`,
				section.conflicts.length && `${section.conflicts.length} with a different URL`,
				section.skipped.length && `${section.skipped.length} skipped`,
			].filter(Boolean);
			addLine(block, `${SECTION_LABELS[key] ?? key} models: ${counts.join(", ")}`, "font-weight: 600;");
			if (section.conflicts.length) {
				addList(block, section.conflicts.map((c) => `${c.name}: mine ${c.url}, imported ${c.imported_url}`));
			}
			if (section.offline.length) {
				addLine(block, `New offline models only work if their files are already on this computer: ${section.offline.join(", ")}`, muted);
			}
			if (section.skipped.length) {
				addLine(block, "Skipped:", muted);
				addList(block, section.skipped.map((s) => `${s.name ?? "(entry without a name)"}: ${s.reason}`));
			}
			conflictCount += section.conflicts.length;
			summary.appendChild(block);
		}
		if (preview.skipped_sections.length) {
			const block = document.createElement("div");
			addLine(block, "Skipped parts of the file:", muted);
			addList(block, preview.skipped_sections.map((s) => (s.key ? `${s.key}: ${s.reason}` : s.reason)));
			summary.appendChild(block);
		}

		const form = dialog.querySelector("form");
		const importButton = dialog.querySelector('button[type="submit"]');
		const nothing = dialog.querySelector(".ondemand-nothing");
		// Set display rather than the hidden attribute, which the fieldset's inline display: flex would override
		dialog.querySelector(".ondemand-conflicts").style.display = conflictCount ? "flex" : "none";
		if (conflictCount) {
			// Here nothing changes only because "Keep mine" ignores the conflicts, so point that out
			nothing.replaceChildren();
			const icon = document.createElement("span");
			icon.textContent = "⚠ ";
			icon.setAttribute("aria-hidden", "true");
			nothing.append(icon, `Nothing would be added or changed. Choose "Use imported" to update the ` +
				`${conflictCount === 1 ? "model" : `${conflictCount} models`} with a different URL.`);
			nothing.style.color = "var(--warning-text, #f0b400)";
		}
		const useImported = () => form.elements.conflicts?.value === "imported";
		const update = () => {
			const changes = useImported() ? preview.changes.imported : preview.changes.keep;
			importButton.disabled = changes === 0;
			nothing.hidden = changes !== 0;
		};
		form.addEventListener("change", update);
		update();

		form.addEventListener("submit", (e) => {
			e.preventDefault();
			finish(useImported());
		});
		dialog.querySelector('button[value="cancel"]').addEventListener("click", () => finish(null));
	});
}

// Menu items for a loader node with url fields: export its sections, export everything, import
function getTransferMenuItems(node) {
	const links = node.constructor.ondemandUrlLinks;
	if (!links?.length) return [];
	const keys = [...new Set(links.map((link) => link.configKey))];
	return [
		null,
		...keys.map((key) => ({ content: `Export ${SECTION_LABELS[key] ?? key} models…`, callback: () => exportModels([key]) })),
		{ content: "Export all models…", callback: () => exportModels(null) },
		{ content: "Import models…", callback: () => importModels() },
	];
}

// When a run adds a model to config.json, offer it in the dropdowns of matching nodes
api.addEventListener("ondemand_loaders.config_updated", ({ detail }) => {
	for (const node of app.graph?.nodes ?? []) {
		for (const link of node.constructor.ondemandUrlLinks ?? []) {
			if (link.configKey !== detail.key) continue;
			const { nameWidget } = getLinkWidgets(node, link);
			if (nameWidget) addComboValue(nameWidget, detail.name);
		}
	}
});



app.registerExtension({
	name: "comfy.francarl.onDemandLoader",
	getNodeMenuItems(node) {
		return getTransferMenuItems(node);
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const urlLinks = getUrlLinks(nodeData);
		if (urlLinks.length > 0) {
			nodeType.ondemandUrlLinks = urlLinks;
			setupUrlWidgets(nodeType, urlLinks);
		}

		if (nodeData.name === "OnDemandCivitaiLikedLoraLoader") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				const loraNameWidget = this.widgets.find((w) => w.name === "lora_name");

				if (loraNameWidget) {

					loraNameWidget.callback = (value) => {
						onLoraChanged(this, value);
					};

					loraNameWidget.searchableCombo = new SearchableCombo();

					loraNameWidget.onClick = (e) => {
						console.log("[ComfyUI-OnDemand-Loaders] Lora Name:", loraNameWidget.value);

						loraNameWidget.searchableCombo.show(loraNameWidget.options.values, {
							event: e.e,
							title: 'Select Preset',
							currentMode: 'list',
							initialExpanded: false,
							onExpandedChange: (isExpanded) => {
								console.log('[ComfyUI-OnDemand-Loaders] SearchableCombo expanded:', isExpanded);
							},
							callback: (selectedItem) => {
								loraNameWidget.value = selectedItem;
								onLoraChanged(this, selectedItem);
							}
						});
					};
				}

				const filterWidget = this.widgets.find((w) => w.name === "filter");
				if (filterWidget) {
					filterWidget.callback = (value) => {
						onFilterChanged(this, value);
					};
				}
				
				this.addWidget("button", "ℹ️ Lora Info", "", () => {
					window.showSelectedLoraInfo(this, loraNameWidget);
				}, { serialize: false });
				
			};
		}
	},
	async setup() {
        window.showSelectedLoraInfo = async (node, widget) => {
            onLoraChanged(node, widget.value);
        };
    }
});