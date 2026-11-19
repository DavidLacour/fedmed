{{- define "fedmed.labels" -}}
app.kubernetes.io/name: fedmed
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{- define "fedmed.securityContext" -}}
runAsNonRoot: true
runAsUser: 10001
runAsGroup: 10001
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities: { drop: ["ALL"] }
seccompProfile: { type: RuntimeDefault }
{{- end }}

{{- define "fedmed.coordinatorUrl" -}}
{{- if .Values.coordinator.externalUrl -}}
{{ .Values.coordinator.externalUrl }}
{{- else -}}
https://{{ .Release.Name }}-coordinator:{{ .Values.coordinator.service.port }}
{{- end -}}
{{- end }}
