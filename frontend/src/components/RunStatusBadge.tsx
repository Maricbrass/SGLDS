import type { ApiStatus } from "../api/types";

type Props = {
  status?: ApiStatus | string;
};

export default function RunStatusBadge({ status }: Props) {
  const value = status || "pending";
  const className = `status-pill status-${value}`;

  return <span className={className}>{value}</span>;
}
